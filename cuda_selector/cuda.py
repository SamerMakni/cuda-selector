import ctypes
import platform
import subprocess
import warnings

def is_mps_available():
    if platform.system() != 'Darwin':
        return False 

    metal = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/Metal.framework/Metal')

    metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    mps_device = metal.MTLCreateSystemDefaultDevice()

    return mps_device != 0

def auto_cuda(criteria='memory', n=1, fallback=True, exclude=None, thresholds=None, sort_fn=None):
    """
    Selects the optimal CUDA device based on specified criteria (memory, power, utilization, temperature) 
    or a custom ranking function, with options to exclude certain devices, apply thresholds, and choose 
    fallback behaviors for macOS devices.

    Parameters:
    -----------
    criteria : str, optional, default='memory'
        The primary selection criterion for the optimal device. 
        Options: 
            - 'memory': selects the device with the most free memory.
            - 'power': selects the device with the lowest power draw.
            - 'utilization': selects the device with the lowest GPU utilization.
            - 'temperature': selects the device with the lowest temperature.
        
    n : int, optional, default=1
        The number of devices to return. If n > 1, the top n devices based on the selection criterion or 
        custom ranking function will be returned as a list.

    fallback : bool, optional, default=True
        Whether to fall back to the CPU if no suitable CUDA device is found. If False and no device is found, 
        a RuntimeError will be raised.

    exclude : list or set of int, optional, default=None
        A list or set of GPU indices to exclude from selection. Excluded devices will not be considered for 
        selection.

    thresholds : dict, optional, default=None
        A dictionary where keys are criteria ('power', 'utilization', or 'temperature') and values are the 
        corresponding thresholds. If a device exceeds the threshold for any of these criteria, it will be excluded 
        from selection.

    sort_fn : callable, optional, default=None
        A custom ranking function to use for sorting the devices. This function should take a dictionary (representing 
        a device) and return a numerical value. Devices will be sorted in ascending order of this value. 
        If not provided, the function will use the default sorting based on the selected criterion.

    Returns:
    --------
    str or list of str
        If `n` is 1, returns a string representing the optimal CUDA device in the form 'cuda:<index>'.
        If `n` is greater than 1, returns a list of strings, each representing an optimal CUDA device 
        (e.g., ['cuda:0', 'cuda:1']).
        If no suitable device is found, returns 'cpu' (or ['cpu'] if `n` > 1).

    Raises:
    -------
    RuntimeError
        If no suitable CUDA device is found and `fallback` is set to False on macOS.

    Warns:
    ------
    UserWarning
        If no suitable CUDA device is found or if there are any warnings related to device availability.

    Notes:
    ------
    - This function uses the `nvidia-smi` command to query GPU information and relies on its output to gather 
      data about memory, power usage, GPU utilization, and temperature.
    - On macOS, if MPS (Multi-Process Service) is available, the function will prioritize the MPS device. 
      If MPS is not available and fallback is not enabled, it will raise an exception.
    """
    exclude = set(exclude) if exclude else set()
    thresholds = thresholds or {} 

    if platform.system() == 'Darwin':
        if is_mps_available():
            warnings.warn("MPS device detected and selected.")
            return "m__inps"
        else:
            if not fallback:
                raise RuntimeError("No MPS device available on macOS.")
            warnings.warn("No MPS device available on macOS. Using CPU instead.")
            return "cpu"
    try:

        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free,memory.total,power.draw,utilization.gpu,temperature.gpu,index',
             '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            encoding='utf-8'
        )
        
        lines = result.stdout.strip().split('\n')
        devices = []

        for line in lines:
            values = [value.strip() for value in line.split(',')]
            
            memory_free = int(values[0])  # Memory free
            memory_total = int(values[1])  # Total memory
            power_draw = float(values[2])  # Power draw
            utilization = float(values[3].rstrip('%'))  # Utilization percentage
            temperature = float(values[4])  # GPU temperature
            index = int(values[5])  # GPU index
            
            # Skip excluded GPUs
            if index in exclude:
                continue
            
            memory_usage = memory_total - memory_free

            device = {
                'memory_free': memory_free,
                'memory_usage': memory_usage,
                'power_draw': power_draw,
                'utilization': utilization,
                'temperature': temperature,
                'index': index
            }

            if any(
                key in thresholds and device[key] > thresholds[key] if key in ['power', 'utilization', 'temperature']
                else key in thresholds and device[key] < thresholds[key] for key in thresholds
            ):
                continue

            devices.append(device)

        if not devices:
            warnings.warn("No suitable CUDA devices found. Using CPU instead.")
            return "cpu" if n == 1 else ["cpu"]

        default_sort_key = {
            'memory': lambda x: -x['memory_free'],  # More free memory is better
            'power': lambda x: x['power_draw'],  # Lower power draw is better
            'utilization': lambda x: x['utilization'],  # Lower utilization is better
            'temperature': lambda x: x['temperature'],  # Lower temperature is better
        }.get(criteria)

        devices.sort(key=sort_fn if sort_fn else default_sort_key)

        sorted_devices = devices[:n]

        return [f'cuda:{d["index"]}' for d in sorted_devices] if n > 1 else f'cuda:{sorted_devices[0]["index"]}'

    except FileNotFoundError:
        warnings.warn("'nvidia-smi' not found. No CUDA devices detected. Using CPU instead.")
        return "cpu" if n == 1 else ["cpu"]