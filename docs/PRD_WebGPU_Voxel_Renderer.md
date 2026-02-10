# **Product Requirements Document: WebGPU Geospatial Voxel Visualizer**

## **1\. Executive Summary**

This project aims to build a high-performance **Geospatial & Simulation Visualization Tool** using **TypeScript** and **WebGPU**. The renderer is designed to visualize large-scale, time-series volumetric data (e.g., weather simulations, urban growth, sensor measurements).

Unlike game engines, this tool is architected as a **standalone, framework-agnostic library**. It can be embedded into any web application to provide scientific visualization capabilities. It utilizes the "Majercik Ray-Box" intersection algorithm to render dynamic, oriented voxels that can transition smoothly between states (position, color, size).

## **2\. Technical Architecture & Core Principles**

### **2.1 Rendering Philosophy (Majercik 2018 \+ Time Series)**

* **No Meshing:** Voxels are rendered as ray-traced bounding boxes.  
* **Smooth Transitions:** All voxel attributes (position, color, size) are interpolated on the GPU. The visualizer renders the state between Time ![][image1] and Time ![][image2].  
* **Geospatial Precision:** Uses **Relative-To-Eye (RTE)** coordinates to handle large world coordinates (e.g., UTM or ECEF) without floating-point jitter.

### **2.2 Library Architecture (TypeScript)**

To ensure the renderer is embeddable in any web application (React, Vue, Svelte, or Vanilla JS), it is built as a framework-agnostic **TypeScript** library.

* **Language:** TypeScript 5.0+  
* **Dependencies:** Zero runtime dependencies (except standard @webgpu/types).  
* **Distribution:** NPM Package (ESM/CJS).

#### **Public API Surface (Draft)**

The library exposes a minimal, high-level API to the host application:

export class VoxelRenderer {  
    // Initialization with a target canvas  
    constructor(canvas: HTMLCanvasElement, options?: RendererOptions);

    // Lifecycle  
    async initialize(): Promise\<void\>;  
      
    // Dependency Injection for Data Source (Adapter Pattern)  
    setDataSource(adapter: IVoxelDataSource): void;

    // Time-Series Control  
    // Host app drives the timeline; renderer handles interpolation.  
    setTime(timestamp: number): void; 

    // Camera Control   
    // Host app can drive the camera (e.g., from Mapbox/Cesium) or let the library handle it.  
    updateCamera(viewMatrix: Float32Array, projectionMatrix: Float32Array, positionHigh: Float64Array): void;

    // Resize handling  
    resize(width: number, height: number): void;

    // Cleanup  
    dispose(): void;  
}

### **2.3 WebGPU Pipeline Architecture**

#### **A. Data Structure (Interpolated Storage)**

To support smooth animation without re-uploading data every frame, each voxel stores a "Start" and "End" state.

* **VoxelInstanceBuffer (Storage Buffer \- ReadOnly):**  
  * **State A (Start):**  
    * vec3\<f32\> pos\_a (Relative to Chunk Origin)  
    * u32 color\_a (Packed RGBA8 or Value ID)  
    * f32 size\_a (Half-extent \- **varies by LOD**)  
  * **State B (Target):**  
    * vec3\<f32\> pos\_b  
    * u32 color\_b  
    * f32 size\_b  
  * **Meta:**  
    * quat\<f32\> rotation (Shared, or interpolated if necessary)  
* **Uniforms:**  
  * float interpolation\_factor (0.0 to 1.0): Controls the blend between State A and State B.  
  * vec3\<f32\> chunk\_world\_origin: High-precision offset for RTE rendering.

#### **B. The Render Pipeline (GPURenderPipeline)**

**1\. Vertex State (Input Assembly)**

* **Logic:**  
  * Fetch State A and State B based on instance\_index.  
  * **Interpolate:** current\_pos \= mix(pos\_a, pos\_b, interpolation\_factor).  
  * **Interpolate:** current\_size \= mix(size\_a, size\_b, interpolation\_factor).  
  * **Interpolate:** current\_color \= mix(color\_a, color\_b, interpolation\_factor).  
  * **Culling:** If current\_size \< EPSILON, move vertex to (0,0,0) to cull degenerate voxels (handling appearing/disappearing data).  
  * Compute Screen-Space AABB for the *interpolated* box.

**2\. Fragment State (Ray-Casting)**

* **Logic:**  
  * Perform Ray-Box Intersection on the *interpolated* box geometry.  
  * Write accurate Depth.  
  * Output current\_color.

## **3\. Streaming & Dynamic Updates (Backend Agnostic)**

### **3.1 Data Source Adapter Pattern (TypeScript)**

To support various backends (e.g., Hierarchical Octrees, Flat Files, Live Simulations), the renderer is decoupled from data fetching via a **TypeScript Interface**. The host application is responsible for implementing this interface or using a provided default.

export interface VoxelDataChunk {  
    id: string;  
    lodLevel: number;  
    worldPosition: Float64Array; // Origin of the chunk  
    data: ArrayBuffer; // Binary data matching GPU struct layout  
}

export interface IVoxelDataSource {  
    // Metadata for the renderer setup  
    getMetadata(): Promise\<{  
        worldBounds: Float64Array;  
        maxLodDepth: number;  
        coordinateSystem: 'cartesian' | 'geospatial';  
    }\>;

    // Request data for a specific region and time  
    requestChunk(  
        bbox: Float64Array,   
        lodLevel: number,   
        timeIndex: number  
    ): Promise\<VoxelDataChunk\>;  
}

* **Implementation A: OctreeBackendAdapter**  
  * Connects to the 64x64x64 Hierarchical Octree DB.  
  * Translates the renderer's requestChunk calls into backend queries.  
  * **LOD Handling:** Maps renderer LOD request (0-6) to octree depth queries (e.g., LOD 0 \= 64³, LOD 6 \= 1 voxel/chunk).  
* **Implementation B: SimulationAdapter**  
  * Connects to live simulation sockets.  
  * Can return raw data arrays directly.

### **3.2 LOD-Aware Memory Management (Dynamic Block Allocator)**

Since different LOD levels produce vastly different voxel counts (LOD 6 \= 1 voxel, LOD 0 \= \~262k voxels), a fixed-slot system is inefficient. We use a **Dynamic Heap Allocator** for GPU memory.

#### **A. Memory Layout**

* **Global Pool:** One massive VoxelInstanceBuffer (e.g., 256MB).  
* **Management:** A CPU-side FreeListAllocator or BuddyAllocator tracks free ranges in the GPU buffer.

#### **B. The Loading Process (LOD Change)**

1. **Camera Move:** Renderer detects Chunk X is now far away.  
2. **LOD Decision:** Switch Chunk X from LOD 0 (High Res) to LOD 3 (Low Res).  
3. **Fetch:** IVoxelDataSource.requestChunk(X, LOD=3).  
4. **Re-Allocate:**  
   * CPU calculates size needed for LOD 3 data.  
   * Allocates a new range in the GPU Heap.  
   * De-allocates the old LOD 0 range (marking it free for others).  
5. **Upload:** queue.writeBuffer to the new range.  
6. **Update Draw List:** Update the IndirectDrawBuffer to point to the new range and instance count.

### **3.3 Time-Series Updates (The "Keyframe" Model)**

This handles updates *within* a specific LOD level (e.g., values changing over time).

1. **Initial State (![][image3]):** Buffer is populated with State A \= Data\_0, State B \= Data\_0.  
2. **Transition (![][image4]):**  
   * Backend sends Data\_1 (must match topology of current LOD).  
   * **Compute Shader (Diff Update):** A Compute Shader (or CPU loop) copies State B to State A (making the target the new start).  
   * **Upload:** Write Data\_1 into State B.  
   * **Animate:** Client increments interpolation\_factor from 0.0 to 1.0 over ![][image5] seconds.

## **4\. Shading & Geospatial Visualization**

### **4.1 Transfer Functions**

Scientific data often arrives as raw values (temperature, density) rather than colors.

* **Approach:** Instead of u32 color, store f32 value.  
* **Colormap Texture:** Bind a 1D Texture (Gradient) to the fragment shader.  
* **Lookup:** final\_color \= textureSample(colormap, sampler, interpolated\_value).  
* This allows changing the visualization theme (Heatmap vs. Coolwarm) instantly without re-uploading geometry.

### **4.2 Relative-To-Eye (RTE) Rendering**

* **Problem:** Standard 32-bit floats jitter at \>10km from origin.  
* **Solution:**  
  * Store voxel positions relative to their **Chunk Origin**.  
  * CPU tracks CameraHigh (double) and ChunkHigh (double).  
  * Uniform RelativeCameraPos \= (CameraHigh \- ChunkHigh).  
  * Shader calculates VertexWorldPos \= VoxelPos \- RelativeCameraPos.

## **5\. Development Roadmap**

### **Phase 1: The "Interpolated Cube"**

* **Goal:** A single cube that smoothly changes size and color.  
* **Tasks:**  
  * Setup TypeScript project structure (Rollup/Vite).  
  * Implement VoxelRenderer class shell.  
  * Implement VoxelState struct and vertex shader interpolation.

### **Phase 2: The Adapter & Allocator**

* **Goal:** Load variable-sized chunks via an interface.  
* **Tasks:**  
  * Define IVoxelDataSource interface.  
  * Implement DynamicBlockAllocator (TS) for the GPU buffer.  
  * Test loading a "Dense" chunk (1000 voxels) and a "Sparse" chunk (1 voxel) into the same buffer.

### **Phase 3: Geospatial Context & Octree Integration**

* **Goal:** Connect to the Octree Backend.  
* **Tasks:**  
  * Implement OctreeBackendAdapter.  
  * Implement RTE Coordinate system.  
  * Visualization of LOD switching (debug colors for LOD levels).

### **Phase 4: Integration**

* **Goal:** Hybrid Rendering.  
* **Tasks:**  
  * Render a textured mesh (e.g., a map base layer) behind the voxels.  
  * Verify depth testing between voxels and terrain.

## **6\. Implementation References (WGSL)**

### **Interpolated Vertex Logic**

struct VoxelInstance {  
    pos\_a: vec3\<f32\>,  
    color\_a: u32,  
    size\_a: f32, // Size depends on LOD level  
    pos\_b: vec3\<f32\>,  
    color\_b: u32,  
    size\_b: f32,  
}

// ... Vertex Logic remains similar, handles varying sizes automatically ...

## **7\. Open Questions**

1. **LOD Popping:** Switching from 1 big voxel to 8 small voxels can be visually jarring.  
   * *Mitigation:* **Per-Voxel Interpolation.** The renderer ignores chunk boundaries for interpolation. By tracking unique voxel IDs/coordinates, we will interpolate attributes individually. This allows us to handle transitions (morphing 1 parent to 8 children) at the voxel level regardless of which chunk they technically belong to.  
2. **Bandwidth:** Streaming "Next State" for 1M voxels at 60Hz is \~32MB/frame (PCIe bottleneck).  
   * *Clarification:* We will not receive full updates every frame. The backend will stream sparse deltas (a stream of changes), significantly reducing bandwidth usage compared to full snapshots. Additionally, these data updates will occur at a rate significantly slower than the render framerate (e.g., 1Hz \- 10Hz), with the GPU handling smooth interpolation in between frames.  
3. **Topology Mismatch:** What if the Octree structure changes between Time T0 and T1?  
   * *Strategy:* **Coordinate-Based Interpolation.** Similar to the LOD mitigation, we do not rely on fixed chunk topology. We map "Start" voxels to "End" voxels based on spatial coordinates/indices. If the structure changes (split or merge), the GPU interpolates between the overlapping coordinate states, treating voxels as independent entities rather than chunk-bound geometry.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAaCAYAAACHD21cAAAA0ElEQVR4XmNgGAX4gYqKCru8vPx/YrGsrKwUWCOQEwXEfQoKCvZSUlKyQCFGqNh3aWlpYZgFQHlzoNgbGB+kcTmcAwVycnJ1QPEtWMQPghlAyUAgZwGypJKSEj/UWUbI4iAAFMsBM4CK5GRkZKSRJYFO8gdpBDKZkMVBABgeouhicADyL1QjaQCo6RzJGoEaJKH+I1njVKjGRehyeAFQw3WQRmBIp6HL4QTA0FSAORMY0qro8rgAE9CWeqjG++iSWAFQ4XOYTWj4LrraUTC8AAB2ckI2bNffOwAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC8AAAAZCAYAAAChBHccAAABf0lEQVR4Xu2VPUvDQBzGT6HgC+gipOTtYgzibAZd/QLStYt+AREcunYS0fqyuzqIi5ubk4tfQnRxql/AQUp9jp5w9/dsmjTBKveDh8Lz/C/3cEkaxiyW/0sURRvQJvUnHs75HvQWhuERzViapjWE/VGFE9il1/gJzM5ADeqPAvbqyD2fxa+xPIIrgycWvageStSFnyTJguoPA2sirHmkfh6w/lCWP9YCcTIIuprJBuWRXaqe53k+/CfVywLzy1WWbyC41kzGpmX5pmrC24JuVS8LbBhXVj6O49B13UD1MLguy9dVH3NLmF9VvSyCIFiprLwJDN5BF9QvQpnlcZgnNNPwfX8Wg+/YdJtmWYgNcurbu2aCy/JQh2YafPBc9x3HmadZEco8eeiUZhp4rtpikPpFKbn8Gc00UP5hgsuf0+yLKRTfl0MfNCxKGeXR60b2usdLu6aFMF9lSNXDX+OiNpyTccqj6IGhk1CLzlbCOOV/HXzY5nDbd6hvsVgsf4dPq9CWNVXcNF0AAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAaCAYAAABVX2cEAAABR0lEQVR4Xu2Svy4FURDGr5AohXL/nd21pVoiCu+ATiGId1BoSHQIUakUIpF4Ar03UIhCJdGLSsVvYmeNufbGJpLb3C+Z7JlvZr6zc2Z6vRGGj6qqJkMIH3+1siynvEYDEtawozzPl6IoSqHGOG/AvdpC/AXs+bvyF2RZdu05ivaxG8+Te+u5BhQsk3BhuSRJZqSdNE3nLC+A3/Jcg6IoAsWx5RBfFTHLKeQizw0EQmdtYp2B0P2/iDG9TITaxJjyDrErnuKE7/HANSH5XIT8UBQipmfee57cNxv/AYJP9Z9t+pgsN5dsq8+0Z9s6kElV2qK06+NxHCeyzOrrk9gcxTiJe7XYow8K8i+sqy8r1ScG8aJ/5OzB5tHidDDty472iXUBxbt6RnwR/93GO4HiO3M+wA5tvBMoXuHdTvle1usz4XNGGBI+AYeEXm1jstBnAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEQAAAAZCAYAAACIA4ibAAACmklEQVR4Xu2WPWgUURDHkxBBJIUYT8N97d6HooVoPAVRRMFSCy3EVlOopFBQtBNsYqEQIURBQUHsBD9AECFFQMHCIqWFoI0iSAq1UNAU8Tf4VmaHze3duYk5eD8Ybuc/897Om9339np6PB6Px5M5fUEQzLdhE3aC/w01jSTUuaDVarV1do6/kHA3QZOB33O53IDRP4RhuE9rywGpFdusNep8Ievg97LWy+XySe3HaDQaKxj02eoyEQOfJ+izxWJxjdWzhPvusVoa1DWpfZqwGm1O1lEqlfbqGNqo9mNw80MMfmB115CLWisUCoPo77S2GFDPNqulUalUtmqfOY65t/wLbr+Osa7r2o9RrVbLYlpj4UXX2R1a5yYr0bdrbbGg6ANWawdqveUa8jghtslqTWHAvaS3ZilhIfexaZ78ehtrBdeMefvmdIJ8cWZpyGkbENCHiD/E7nB9ysYzpJ973MQ+Bs32/AK4hnyyegTn4Bbit+v1es7GYpA0LJMxYIONuQN4htf5qPg05DV22OZZ2Ho7GTfVgb2UWrD3ds40ZJy86VYX5Cgg/khy5Hiw8Rgs9owkWl1APyix6FPM9Rj5T2xeFoR/zqtn2Lg8CBtPodc15IQNRBA/Ijn5fL5kYzFIetqkIZewn8o/h/3gsk+lZYJrxjWrtwKN2CVr4GFVbSyipYYw0XFJatIQ2dNflT+aduNO4cu30WqtIGcCNb3CbtiYRra61J64ZVjQ26gR1vQpTd4VtG/KP+smHYy0LJAzx2pp2Lq1hQlfzMC9IfYvR1swwXlsTvkXsF9c9qq0fyZYgv86WTVkt0zCvlvr/KvYlM3rBqKGsAMCG2sLmrGKicaxaWzExrsBttB+an8jDcFmuvWhejwez7LnN24C3B4A+dd/AAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAZCAYAAAA8CX6UAAABL0lEQVR4XmNgGAUDA+Tl5WuA+D8SDkKTd0KT/6+goJCBrAYFyMnJbQAqqAQpBLKt0eWB4llAPA9dHB2wABW9l5GREYLauhxdAVBsCRBHo4ujAKALbID4IIgNc76SkpIashqg2DOgiyWQxTAAUFEtCEPZx6GGTYPJKyoq6gP5ZxA6sABxcXFuoKLXQCYzVAjkzXsgw4CurAcJANlbga5xR+jCAoCKvIF4PZpYMdRVL1VUVNiB9AcQjawGAwAV9QBtzkUWA4YPP0gzyDBZWVk3IL0NWR4rACo6B4wtXSzivVBXdQNxKbo8CgAGojhQ0RN0cRAAiisC8V8g/gTERujycADyMzAATwEV7Qd6TQtdHgSAclZAvBhdHA6AkoZQZyPjEnR1IAC0JARdbBSMAmoCALGWUXhSnDamAAAAAElFTkSuQmCC>