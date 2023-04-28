import multiprocessing as mp
mp.set_start_method('spawn', force=True)
from multiprocessing import Process, Queue

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50

from torch2trt import torch2trt

import numpy as np

import time

from tqdm import tqdm

def compile_model(batch_size, device, input_shape=(3,224,224), checkpoint=None):
    with torch.no_grad():
        original_model = resnet50().cuda(device=f'cuda:{device}').eval()
        if checkpoint is not None:
            ckp = torch.load(checkpoint, map_location=f'cuda:{device}')
            original_model.load_state_dict(ckp)        
        st = time.time()
        print("Converting")
        model_trt = torch2trt(original_model, [torch.randn(batch_size,*input_shape).cuda(device=f'cuda:{device}')], 
            fp16_mode=True, use_onnx=True, max_batch_size=batch_size)
        print(f"Done in {time.time()-st:0.5f}sec")
    del original_model
    return model_trt
    
class dummy_dataset(Dataset):
    def __init__(self):
        super().__init__()
    
    def __getitem__(self, index):
        return torch.randn(3, 224, 224), torch.randint(0, 1000, (1,)).item()
        
    def __len__(self):
        return 10240

def run_single_inference():
    dataset = dummy_dataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=16)
    model = compile_model(batch_size=32, device=0)
    model.eval()
    st = time.time()
    for data, target in tqdm(dataloader):
        _ = model(data.cuda(device=0))
    del model
    print(f"Done in {time.time()-st:0.5f}sec")
    
    
def batch_consumer(device, batch_size, data_queue, result_queue, compile_monitor):
    torch.cuda.set_device(device)
    model = compile_model(batch_size=batch_size, device=device)
    compile_monitor.put(True)
    while True:
        inp = data_queue.get()
        if inp is None:
            break
        out = model(inp.cuda(device=f'cuda:{device}')).cpu()
        result_queue.put(out)
    
class DataParallelTRT:
    def __init__(self, device_indices, total_batch_size=None, batch_size_list=None):
        assert total_batch_size is not None or batch_size_list is not None, "Either total_batch_size or batch_size_list should be provided"
        if total_batch_size is not None and batch_size_list is not None:
            print("Both total_batch_size and batch_size_list are provided. batch_size_list will be used.")
        parallel_list =[]
        data_queue_list = []
        result_queue_list = []
        compile_monitor = Queue()
        from multiprocessing import Manager
        self.model_list = {}
        if batch_size_list is None:
            batch_size_list = [total_batch_size//len(device_indices)]*len(device_indices)
            
        parallel_list = []
        for index in device_indices:
            data_queue_list.append(Queue())
            result_queue_list.append(Queue())
            parallel_list.append(Process(target=batch_consumer, args=(index, batch_size_list[index], data_queue_list[-1], result_queue_list[-1],compile_monitor,)))
            parallel_list[-1].start()
            
        for _ in range(len(device_indices)):
            compile_monitor.get()
         
        self.batch_index_list = [0, *np.cumsum(batch_size_list)]
        self.data_queue_list = data_queue_list
        self.result_queue_list = result_queue_list
        self.parallel_list = parallel_list

        
    def forward(self, x):
        # x: (batch_size, ...)
        for idx, data_queue in enumerate(self.data_queue_list):
            data_queue.put(x[self.batch_index_list[idx]:self.batch_index_list[idx+1]])
        output = []
        for result_queue in self.result_queue_list:
            output.append(result_queue.get())
        return torch.cat(output, dim=0)
            
    def __del__(self):
        for data_queue in self.data_queue_list:
            data_queue.put(None)
        for parallel in self.parallel_list:
            parallel.join()
    
    def __call__(self, x):
        return self.forward(x)
            
def run_multi_inference():
    dataset = dummy_dataset()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=16)
    model = DataParallelTRT(device_indices=[0,1,2,3], batch_size_list=[32,32,32,32])
    st = time.time()
    for data, target in tqdm(dataloader):
        _ = model(data)
    for data_queue in model.data_queue_list:
        data_queue.put(None)
    print(f"Done in {time.time()-st:0.5f}sec")
    
    del model
    

def batch_consumer_dist(device, batch_size, data_queue, result_queue, compile_monitor):
    from torch.utils.data.distributed import DistributedSampler
    torch.cuda.set_device(device)
    model = compile_model(batch_size=batch_size, device=device)
    dataset = dummy_dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=DistributedSampler(dataset, num_replicas=4, rank=device))
    compile_monitor.put(True)
    pbar = tqdm(dataloader) if device==0 else dataloader
    for inp, target in pbar:
        _ = data_queue.get()
        out = model(inp.cuda(device=f'cuda:{device}')).cpu()
        result_queue.put(out)
    result_queue.put(None)
    
class DataParallelTRT_DistSample:
    def __init__(self, device_indices, total_batch_size=None, batch_size_list=None):
        assert total_batch_size is not None or batch_size_list is not None, "Either total_batch_size or batch_size_list should be provided"
        if total_batch_size is not None and batch_size_list is not None:
            print("Both total_batch_size and batch_size_list are provided. batch_size_list will be used.")
        parallel_list =[]
        data_queue_list = []
        result_queue_list = []
        compile_monitor = Queue()
        
        self.model_list = {}
        if batch_size_list is None:
            batch_size_list = [total_batch_size//len(device_indices)]*len(device_indices)
            
        parallel_list = []
        for index in device_indices:
            data_queue_list.append(Queue())
            result_queue_list.append(Queue())
            parallel_list.append(Process(target=batch_consumer_dist, args=(index, batch_size_list[index], data_queue_list[-1], result_queue_list[-1],compile_monitor,)))
            parallel_list[-1].start()
            
        for _ in range(len(device_indices)):
            compile_monitor.get()
         
        self.batch_index_list = [0, *np.cumsum(batch_size_list)]
        self.data_queue_list = data_queue_list
        self.result_queue_list = result_queue_list
        self.parallel_list = parallel_list

        
    def forward(self):
        # x: (batch_size, ...)
        for idx, data_queue in enumerate(self.data_queue_list):
            data_queue.put(idx)
        output = []
        for result_queue in self.result_queue_list:
            out = result_queue.get()
            output.append(out)
        if out is None:
            return None
        return torch.cat(output, dim=0)
    
    def __call__(self):
        return self.forward()
            
def run_multi_inference_sample():
    model = DataParallelTRT_DistSample(device_indices=[0,1,2,3], batch_size_list=[32,32,32,32])
    st = time.time()
    while True:
        out = model()
        if out is None:
            break
    print(f"Done in {time.time()-st:0.5f}sec")
    
    del model


if __name__ == "__main__":
    run_single_inference()
    run_multi_inference()
    run_multi_inference_sample()