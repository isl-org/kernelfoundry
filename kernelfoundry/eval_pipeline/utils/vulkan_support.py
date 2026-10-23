"""Shared Vulkan compute-pipeline helpers for kernelfoundry tasks.

A Vulkan task only differs from another in its compiled shader, buffer count, and
push-constant layout; everything else (instance/device setup, descriptor sets, command
buffer recording, submission) is identical boilerplate. This module builds that boilerplate
once via :func:`create_vulkan_compute_context`, returning a :class:`VulkanCtx` that a task's
``vulkan_ctx`` fixture can expose directly.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import vulkan as vk


@dataclass
class VulkanBuffer:
    """A storage buffer plus its backing device memory and byte size."""

    buffer: Any
    memory: Any
    size: int


@dataclass
class VulkanCtx:
    """Holds the Vulkan objects (device, pipeline, descriptor set, command buffer) needed to
    dispatch a compute shader. Built once per test session via create_vulkan_compute_context().
    """

    device: Any
    queue: Any
    command_buffer: Any
    descriptor_set: Any
    pipeline: Any
    pipeline_layout: Any
    mem_properties: Any
    _recorded_key: Any = field(default=None, init=False, repr=False)

    def find_memory_type(self, type_bits: int, flags: int) -> int:
        """Finds a device memory type index matching `type_bits` and property `flags`."""
        for i in range(self.mem_properties.memoryTypeCount):
            if (type_bits & (1 << i)) and (self.mem_properties.memoryTypes[i].propertyFlags & flags) == flags:
                return i
        raise RuntimeError("No suitable Vulkan memory type found")

    def make_buffer(self, size_bytes: int) -> VulkanBuffer:
        """Allocates a host-visible/coherent storage buffer of the given byte size."""
        buf = vk.vkCreateBuffer(
            self.device,
            vk.VkBufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size=size_bytes,
                usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            ),
            None,
        )
        reqs = vk.vkGetBufferMemoryRequirements(self.device, buf)
        mem_type = self.find_memory_type(
            reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        mem = vk.vkAllocateMemory(
            self.device,
            vk.VkMemoryAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize=reqs.size,
                memoryTypeIndex=mem_type,
            ),
            None,
        )
        vk.vkBindBufferMemory(self.device, buf, mem, 0)
        return VulkanBuffer(buffer=buf, memory=mem, size=size_bytes)

    def upload(self, vbuf: VulkanBuffer, data: np.ndarray) -> None:
        """Copies a numpy array into a storage buffer's host-visible memory."""
        ptr = vk.vkMapMemory(self.device, vbuf.memory, 0, data.nbytes, 0)
        np.copyto(np.frombuffer(ptr, dtype=data.dtype), data)
        vk.vkUnmapMemory(self.device, vbuf.memory)

    def download(self, vbuf: VulkanBuffer, dtype=np.float32) -> np.ndarray:
        """Waits for all submitted work to complete, then reads a storage buffer's
        host-visible memory back into a numpy array.
        """
        self.wait()
        ptr = vk.vkMapMemory(self.device, vbuf.memory, 0, vbuf.size, 0)
        result = np.frombuffer(ptr, dtype=dtype).copy()
        vk.vkUnmapMemory(self.device, vbuf.memory)
        return result

    def _record(
        self,
        buffers: Sequence[VulkanBuffer],
        n: int,
        push_constants: bytes,
        local_size_x: int,
    ) -> None:
        """(Re-)records the command buffer and descriptor set bindings for this dispatch shape.

        A no-op if the last recorded call used the same buffers/shape: within one benchmark
        loop the bindings are static, so re-recording (and re-binding descriptors) on every
        call would be pure CPU-side overhead measured as if it were kernel runtime.
        """
        key = (tuple(vbuf.buffer for vbuf in buffers), n, push_constants, local_size_x)
        if key == self._recorded_key:
            return
        self._recorded_key = key

        writes = [
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=self.descriptor_set,
                dstBinding=i,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[vk.VkDescriptorBufferInfo(buffer=vbuf.buffer, offset=0, range=vbuf.size)],
            )
            for i, vbuf in enumerate(buffers)
        ]
        vk.vkUpdateDescriptorSets(self.device, len(writes), writes, 0, None)

        vk.vkResetCommandBuffer(self.command_buffer, 0)
        vk.vkBeginCommandBuffer(
            self.command_buffer,
            vk.VkCommandBufferBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                # Allows this (unmodified) command buffer to be resubmitted while a previous
                # submission is still pending, so submit() calls can be batched without an
                # intervening wait() — mirrors how an OpenCL in-order queue pipelines enqueues.
                flags=vk.VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
            ),
        )
        vk.vkCmdBindPipeline(self.command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline)
        vk.vkCmdBindDescriptorSets(
            self.command_buffer,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.pipeline_layout,
            0,
            1,
            [self.descriptor_set],
            0,
            None,
        )
        if push_constants:
            push_constants_buf = vk.ffi.new("char[]", push_constants)
            vk.vkCmdPushConstants(
                self.command_buffer,
                self.pipeline_layout,
                vk.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                len(push_constants),
                push_constants_buf,
            )
        vk.vkCmdDispatch(self.command_buffer, math.ceil(n / local_size_x), 1, 1)
        vk.vkEndCommandBuffer(self.command_buffer)

    def submit(
        self,
        buffers: Sequence[VulkanBuffer],
        n: int,
        push_constants: bytes = b"",
        local_size_x: int = 256,
    ) -> None:
        """Records (if needed) and submits ceil(n / local_size_x) workgroups to the queue,
        without waiting for completion. Call wait() (or download(), which waits internally)
        once submissions are batched, to synchronize.
        """
        self._record(buffers, n, push_constants, local_size_x)
        vk.vkQueueSubmit(
            self.queue,
            1,
            [vk.VkSubmitInfo(sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO, pCommandBuffers=[self.command_buffer])],
            None,
        )

    def wait(self) -> None:
        """Blocks until all work submitted to the queue has completed."""
        vk.vkQueueWaitIdle(self.queue)

    def dispatch(
        self,
        buffers: Sequence[VulkanBuffer],
        n: int,
        push_constants: bytes = b"",
        local_size_x: int = 256,
    ) -> None:
        """Binds `buffers` to the pipeline's descriptor set (in order, starting at binding 0),
        pushes `push_constants` if any, and dispatches ceil(n / local_size_x) workgroups,
        blocking until the GPU is done. Convenience for one-off calls; for tight loops, use
        submit() repeatedly and wait() once to avoid a full CPU-GPU sync on every call.
        """
        self.submit(buffers, n, push_constants, local_size_x)
        self.wait()


def create_vulkan_compute_context(
    spv_path: str | Path,
    num_storage_buffers: int,
    push_constant_size: int = 0,
    app_name: str = "kernelfoundry",
) -> VulkanCtx:
    """Builds a VkInstance/VkDevice/compute pipeline for a compiled SPIR-V module.

    Picks the first integrated/discrete GPU with a compute-capable queue family, exactly as
    other kernelfoundry backends pick their first available device.

    Args:
        spv_path: Path to the compiled `.spv` shader module (see
            :func:`kernelfoundry.compiler.compile_glsl_to_spirv`).
        num_storage_buffers: Number of SSBO bindings (0, 1, 2, ...) the shader declares.
        push_constant_size: Byte size of the shader's push_constant block, 0 if none.
        app_name: VkApplicationInfo application name (cosmetic).

    Returns:
        A VulkanCtx ready to receive make_buffer()/upload()/dispatch()/download() calls.
    """
    spv_path = Path(spv_path)
    assert spv_path.exists(), f"{spv_path} not found; did build() run?"
    spv_bytes = spv_path.read_bytes()

    app_info = vk.VkApplicationInfo(
        sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName=app_name,
        apiVersion=vk.VK_API_VERSION_1_0,
    )
    instance = vk.vkCreateInstance(
        vk.VkInstanceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, pApplicationInfo=app_info), None
    )

    phys = None
    for p in vk.vkEnumeratePhysicalDevices(instance):
        props = vk.vkGetPhysicalDeviceProperties(p)
        if props.deviceType in (
            vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
            vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
        ):
            phys = p
            break
    assert phys is not None, "No Vulkan GPU device found"

    queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(phys)
    qfi = next(i for i, q in enumerate(queue_families) if q.queueFlags & vk.VK_QUEUE_COMPUTE_BIT)

    device = vk.vkCreateDevice(
        phys,
        vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pQueueCreateInfos=[
                vk.VkDeviceQueueCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    queueFamilyIndex=qfi,
                    queueCount=1,
                    pQueuePriorities=[1.0],
                )
            ],
            queueCreateInfoCount=1,
        ),
        None,
    )
    queue = vk.vkGetDeviceQueue(device, qfi, 0)
    mem_properties = vk.vkGetPhysicalDeviceMemoryProperties(phys)

    bindings = [
        vk.VkDescriptorSetLayoutBinding(
            binding=i,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=1,
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
        )
        for i in range(num_storage_buffers)
    ]
    descriptor_set_layout = vk.vkCreateDescriptorSetLayout(
        device,
        vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, pBindings=bindings
        ),
        None,
    )
    push_constant_ranges = (
        [vk.VkPushConstantRange(stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=push_constant_size)]
        if push_constant_size
        else []
    )
    pipeline_layout = vk.vkCreatePipelineLayout(
        device,
        vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            pSetLayouts=[descriptor_set_layout],
            pPushConstantRanges=push_constant_ranges,
        ),
        None,
    )
    shader_module = vk.vkCreateShaderModule(
        device,
        vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, codeSize=len(spv_bytes), pCode=spv_bytes
        ),
        None,
    )
    stage = vk.VkPipelineShaderStageCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
        module=shader_module,
        pName="main",
    )
    pipeline = vk.vkCreateComputePipelines(
        device,
        None,
        1,
        [
            vk.VkComputePipelineCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, stage=stage, layout=pipeline_layout
            )
        ],
        None,
    )[0]

    descriptor_pool = vk.vkCreateDescriptorPool(
        device,
        vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=1,
            pPoolSizes=[
                vk.VkDescriptorPoolSize(
                    type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=max(num_storage_buffers, 1)
                )
            ],
        ),
        None,
    )
    descriptor_set = vk.vkAllocateDescriptorSets(
        device,
        vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[descriptor_set_layout],
        ),
    )[0]

    command_pool = vk.vkCreateCommandPool(
        device,
        vk.VkCommandPoolCreateInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, queueFamilyIndex=qfi),
        None,
    )
    command_buffer = vk.vkAllocateCommandBuffers(
        device,
        vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        ),
    )[0]

    return VulkanCtx(
        device=device,
        queue=queue,
        command_buffer=command_buffer,
        descriptor_set=descriptor_set,
        pipeline=pipeline,
        pipeline_layout=pipeline_layout,
        mem_properties=mem_properties,
    )
