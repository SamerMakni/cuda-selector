import ctypes
import platform
import subprocess
import warnings

def is_mps_available():
    """
    Checks if the Multi-Process Service (MPS) is available on the system.
    """
    if platform.system() != 'Darwin':
        return False 
    metal = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/Metal.framework/Metal')
    metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    mps_device = metal.MTLCreateSystemDefaultDevice()
    return mps_device != 0

def auto_cuda(criteria='memory', n=1, fallback=True, exclude=None, thresholds=None, sort_fn=None):
    """
    Selects the optefzfezfezfimal CUDA device based on specified criteria (memory, power, utilization, temperature)
    or a custom ranking function, with options to exclude certain devices, apply thresholds, and choose
    fallback behaviors for macOS devices.

    :param criteria: The primary selection criterion for the optimal device.
                      Options: 'memory', 'power', 'utilization', or 'temperature'.
                      Default is 'memory'.
    :type criteria: str, optional
    :param n: The number of devices to return. Default is 1.
    :type n: int, optional
    :param fallback: Whether to fall back to the CPU if no suitable CUDA device is found.
                     Default is True.
    :type fallback: bool, optional
    :param exclude: A list or set of GPU indices to exclude from selection.
    :type exclude: list or set of int, optional
    :param thresholds: A dictionary of thresholds where the keys are criteria ('power', 'utilization', 'temperature')
                       and the values are the corresponding thresholds. Devices exceeding these thresholds are excluded.
    :type thresholds: dict, optional
    :param sort_fn: A custom ranking function for sorting devices. 
                     The function should take a device dictionary and return a numerical value.
    :type sort_fn: callable, optional

    :returns: If `n` is 1, returns a string representing the optimal CUDA device (e.g., 'cuda:0').
              If `n` is greater than 1, returns a list of strings (e.g., ['cuda:0', 'cuda:1']).
              If no suitable device is found, returns 'cpu' (or ['cpu'] if `n` > 1).
    :rtype: str or list of str
    :raises RuntimeError: If no suitable CUDA device is found and `fallback` is False on macOS.
    :warn UserWarning: If no suitable CUDA device is found or if there are any warnings related to device availability.
    
    :note: 
        This function uses the `nvidia-smi` command to query GPU information and relies on its output 
        to gather data about memory, power usage, GPU utilization, and temperature.
        
        On macOS, if MPS (Multi-Process Service) is available, the function will prioritize the MPS device.
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
            
            memory_free = int(values[0])
            memory_total = int(values[1])
            power_draw = float(values[2])
            utilization = float(values[3].rstrip('%'))
            temperature = float(values[4])
            index = int(values[5])
            
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

            exclude_device = any(
                (key == 'power' and device['power_draw'] > thresholds[key]) or
                (key in ['utilization', 'temperature'] and device[key] > thresholds[key]) or
                (key not in ['power', 'utilization', 'temperature'] and device[key] < thresholds[key])
                for key in thresholds
            )
            if exclude_device:
                continue

            devices.append(device)

        if not devices:
            warnings.warn("No suitable CUDA devices found. Using CPU instead.")
            return "cpu" if n == 1 else ["cpu"]

        default_sort_key = {
            'memory': lambda x: (-x['memory_free'], x['index']),  # Descending memory, then index
            'power': lambda x: (x['power_draw'], -x['memory_free']),  # Ascending power, then descending memory
            'utilization': lambda x: (x['utilization'], -x['memory_free']),  # Ascending utilization, then memory
            'temperature': lambda x: (x['temperature'], -x['memory_free']),  # Ascending temp, then memory
        }.get(criteria)

        if sort_fn:
            devices.sort(key=sort_fn, reverse=True) 
        else:
            devices.sort(key=default_sort_key)
        sorted_devices = devices[:n]

        return [f'cuda:{d["index"]}' for d in sorted_devices] if n > 1 else f'cuda:{sorted_devices[0]["index"]}'

    except FileNotFoundError:
        warnings.warn("'nvidia-smi' not found. No CUDA devices detected. Using CPU instead.")
        return "cpu" if n == 1 else ["cpu"]