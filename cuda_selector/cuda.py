import subprocess

def auto_cuda(criteria='memory'):
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free,memory.total,power.draw,utilization.gpu,index',
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
        utilization = float(values[3].rstrip('%'))  # Utilization percentage as float
        index = int(values[4])  # GPU index
        
        memory_usage = memory_total - memory_free
        
        devices.append({
            'memory_free': memory_free,
            'memory_usage': memory_usage,
            'power_draw': power_draw,
            'utilization': utilization,
            'index': index
        })

    if not devices:
        return f"\\033[93mNo CUDA devices detected. Using CPU instead.\\033[0m"

    if criteria == 'memory':
        optimal_device = max(devices, key=lambda x: x['memory_free'])
    elif criteria == 'power':
        optimal_device = min(devices, key=lambda x: x['power_draw'])
    elif criteria == 'utilization':
        optimal_device = min(devices, key=lambda x: x['utilization'])
    else:
        raise ValueError("Invalid selection criteria. Choose 'memory', 'power', or 'utilization'.")

    return f'cuda:{{optimal_device["index"]}}'
