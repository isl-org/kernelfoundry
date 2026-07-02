from contextlib import contextmanager
import multiprocessing as mp


class ResourceAllocator:
    def __init__(self, device_ids: list, max_inference_requests: int = 10):
        self.device_ids = device_ids
        self.num_gpus = len(device_ids)
        # Create a semaphore to limit total GPU access
        self.gpu_semaphore = mp.Semaphore(self.num_gpus)
        # Track which GPUs are in use
        self.gpu_status = mp.Array("i", [0] * self.num_gpus)
        # Lock for accessing gpu_status
        self.gpu_lock = mp.Lock()

        # Create a semaphore to limit concurrent inference calls
        self.max_inference_requests = max_inference_requests
        self.inference_semaphore = mp.Semaphore(max_inference_requests)
        # Create list of available inference servers
        self.inference_servers = list(range(max_inference_requests))
        self.server_status = mp.Array("i", [0] * max_inference_requests)
        self.inference_lock = mp.Lock()

        print(f"[Resources] ResourceAllocator initialized with GPU device IDs: {device_ids}\
                  and an LLM request limit of {max_inference_requests}")

    def get_available_gpu(self):
        """Find and reserve an available GPU."""
        with self.gpu_lock:
            for i in range(self.num_gpus):
                if self.gpu_status[i] == 0:
                    self.gpu_status[i] = 1
                    return self.device_ids[i]
        return None

    def release_gpu(self, gpu_id: int):
        """Release a GPU back to the pool by device ID."""
        with self.gpu_lock:
            index = self.device_ids.index(gpu_id)
            self.gpu_status[index] = 0

    @contextmanager
    def reserve_gpu(self):
        """Context manager for GPU reservation."""
        self.gpu_semaphore.acquire()
        gpu_id = self.get_available_gpu()
        try:
            yield gpu_id
        finally:
            self.release_gpu(gpu_id)
            self.gpu_semaphore.release()

    def get_available_server(self):
        """Find and reserve an available inference server."""
        with self.inference_lock:
            for i in range(self.max_inference_requests):
                if self.server_status[i] == 0:
                    self.server_status[i] = 1
                    return self.inference_servers[i]
        return None

    def release_server(self, server_id: int):
        """Release an inference server back to the pool by server ID."""
        with self.inference_lock:
            index = self.inference_servers.index(server_id)
            self.server_status[index] = 0

    @contextmanager
    def reserve_server(self):
        """Context manager for inference server reservation."""
        self.inference_semaphore.acquire()
        server_id = self.get_available_server()
        try:
            yield server_id
        finally:
            self.release_server(server_id)
            self.inference_semaphore.release()
