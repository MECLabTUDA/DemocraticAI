from network.model import get_model
from thop import profile
import torch
from torch.utils.flop_counter import FlopCounterMode
import torch.nn as nn
import click
from zeus.monitor import ZeusMonitor
import json

def get_datum(dataset, B: int= 1):
    if dataset in ["fetalAbdominal", "XRayMimic", "XRayMimic200"]:
        return torch.randn(B, 1, 64, 64)
    elif dataset in ["crc"]:
        return torch.randn(B, 3, 96, 96)
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    

def get_flops(model: nn.Module, inp, train):
    model.train(train)
    
    inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)

    flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
    with flop_counter:
        if train:
            model(inp).sum().backward()
        else:
            with torch.no_grad():
                model(inp)
    total_flops = flop_counter.get_total_flops()
    return {"flops": total_flops}

def get_macs_and_params(model, inp):
    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=(inp,), verbose=False)
    return {"macs": macs, "params": params}

def get_peak_mem_consumption(model: nn.Module, inp, train):
    torch.cuda.reset_peak_memory_stats()
    model.train(train)
    if train:
        optimizer = torch.optim.Adam(model.parameters())
        model(inp).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
    else:
        with torch.no_grad():
            model(inp)

    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated()
    return {"mem": mem}

def get_time(model: nn.Module, inp, train):
    model.train(train)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    if train:
        optimizer = torch.optim.Adam(model.parameters())
        for _ in range(100):
            optimizer.zero_grad()
            model(inp).sum().backward()
            optimizer.step()
    else:
        with torch.no_grad():
            for _ in range(100):
                model(inp)
    end.record()

    torch.cuda.synchronize()
    return {"time": start.elapsed_time(end)}

def get_energy_consumption(model: nn.Module, inp, train):
    model.train(train)
    monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])

    monitor.begin_window("epoch")
    if train:
        optimizer = torch.optim.Adam(model.parameters())
        for _ in range(100):
            optimizer.zero_grad()
            model(inp).sum().backward()
            optimizer.step()
    else:
        with torch.no_grad():
            for _ in range(100):
                model(inp)
    mes = monitor.end_window("epoch")

    return {"energy": mes.gpu_energy[0]}

@click.command()
@click.option('--command', 'command', help='Command to run', required=True)
@click.option('--data', 'data_type', help='Data type', required=True)
@click.option('--model', 'model_type', default='mednca', help='Model type', required=True)
@click.option('--mode', 'mode', default='train', help='Mode (train/eval)', required=True)
def main(command, data_type, model_type, mode):
    assert torch.cuda.memory_allocated(0) == 0, f"Expected no memory allocated on GPU 0, but found {torch.cuda.memory_allocated(0)}"
    inp = get_datum(data_type, B=1 if mode=="eval"else 4).cuda()
    model = get_model(data_type, model_type).cuda()
    if command == "macs":
        result = get_macs_and_params(model, inp)
    elif command == "flops":
        result = get_flops(model, inp, mode == "train")
    elif command == "mem":
        result = get_peak_mem_consumption(model, inp, mode == "train")
    elif command == "time":
        result = get_time(model, inp, mode == "train")
    elif command == "energy":
        result = get_energy_consumption(model, inp, mode == "train")
    else:
        raise ValueError(f"Unknown command: {command}")
    print(json.dumps(result))

if __name__ == "__main__":
    main()