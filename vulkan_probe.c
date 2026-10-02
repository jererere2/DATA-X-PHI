#include <vulkan/vulkan.h>
#include <stdio.h>

int main() {
    VkApplicationInfo appInfo = {0};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "DATA Vulkan Wave64 Probe";
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo = {0};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkInstance instance;
    if (vkCreateInstance(&createInfo, NULL, &instance) != VK_SUCCESS) {
        printf("ERROR: No se pudo crear la instancia de Vulkan.\n");
        return 1;
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
    printf("==================================================================\n");
    printf("  SONDA VULKAN WAVE64 // CEREBRO CUANTICO SODAX (LORD AZ12)\n");
    printf("  Dispositivos Vulkan detectados en el bus PCIe: %u\n", deviceCount);
    printf("==================================================================\n");

    if (deviceCount == 0) {
        printf("No se encontraron dispositivos Vulkan.\n");
        vkDestroyInstance(instance, NULL);
        return 1;
    }

    VkPhysicalDevice devices[8];
    if (deviceCount > 8) deviceCount = 8;
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices);

    for (uint32_t i = 0; i < deviceCount; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        printf("\n[GPU %u] %s\n", i, props.deviceName);
        printf("  - Tipo de Dispositivo: %s\n", 
               props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? "GPU Dedicada (Discrete)" :
               props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? "GPU Integrada" : "CPU / Software");
        printf("  - Versión API Vulkan: %u.%u.%u\n", 
               VK_VERSION_MAJOR(props.apiVersion),
               VK_VERSION_MINOR(props.apiVersion),
               VK_VERSION_PATCH(props.apiVersion));

        VkPhysicalDeviceSubgroupProperties subgroupProps = {0};
        subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

        VkPhysicalDeviceProperties2 props2 = {0};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &subgroupProps;

        vkGetPhysicalDeviceProperties2(devices[i], &props2);

        printf("  - Tamaño de Subgroup (Wavefront): %u hilos\n", subgroupProps.subgroupSize);
        if (subgroupProps.subgroupSize == 64) {
            printf("  ⚡ ESTADO: NATIVE WAVE64 ACTIVO! (AMD GCN5 / VEGA ARCHITECTURE)\n");
            printf("  ⚡ Modos de cálculo: Wave64 Lock-Step, Shuffle, Arithmetic, Ballot\n");
        }

        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(devices[i], &memProps);
        for (uint32_t h = 0; h < memProps.memoryHeapCount; h++) {
            if (memProps.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                double vram_gb = (double)memProps.memoryHeaps[h].size / (1024.0 * 1024.0 * 1024.0);
                printf("  - Memoria VRAM Dedicada (HBM2): %.2f GB\n", vram_gb);
            }
        }
    }
    printf("\n==================================================================\n");

    vkDestroyInstance(instance, NULL);
    return 0;
}
