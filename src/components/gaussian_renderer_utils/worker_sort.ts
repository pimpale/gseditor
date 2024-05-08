import { GaussianScene } from "./sceneLoader"

type SortWorkerInput = {
    viewMatrix: Float32Array
    sortingAlgorithm: string
    gaussians: GaussianScene
}

self.onmessage = (event: MessageEvent<SortWorkerInput>) => {
    // Sort gaussians event
    if (event.data.viewMatrix) {
        const { gaussians: gaussians, viewMatrix, sortingAlgorithm } = event.data

        const start = performance.now()

        // Sort the gaussians!
        const depthIndex = sortGaussiansByDepth(gaussians, viewMatrix, sortingAlgorithm)

        // create a new object to store the sorted data
        const data = {
            count: gaussians.count,
            colors: new Float32Array(gaussians.count * 3),
            positions: new Float32Array(gaussians.count * 3),
            opacities: new Float32Array(gaussians.count),
            cov3Da: new Float32Array(gaussians.count * 3),
            cov3Db: new Float32Array(gaussians.count * 3),
        }

        // Update arrays containing the data
        for (let j = 0; j < depthIndex.length; j++) {
            const i = depthIndex[j]

            data.colors[j*3] = gaussians.colors[i*3]
            data.colors[j*3+1] = gaussians.colors[i*3+1]
            data.colors[j*3+2] = gaussians.colors[i*3+2]

            data.positions[j*3] = gaussians.positions[i*3]
            data.positions[j*3+1] = gaussians.positions[i*3+1]
            data.positions[j*3+2] = gaussians.positions[i*3+2]

            data.opacities[j] = gaussians.opacities[i]

            // Split the covariance matrix into two vec3
            // so they can be used as vertex shader attributes
            data.cov3Da[j*3] = gaussians.cov3Da[i*3]
            data.cov3Da[j*3+1] = gaussians.cov3Da[i*3+1]
            data.cov3Da[j*3+2] = gaussians.cov3Da[i*3+2]

            data.cov3Db[j*3] = gaussians.cov3Db[i*3]
            data.cov3Db[j*3+1] = gaussians.cov3Db[i*3+1]
            data.cov3Db[j*3+2] = gaussians.cov3Db[i*3+2]
        }

        const sortTime = `${((performance.now() - start)/1000).toFixed(3)}s`
        console.log(`[Worker] Sorted ${gaussians.count} gaussians in ${sortTime}. Algorithm: ${sortingAlgorithm}`)

        postMessage({
            data, sortTime,
        })
    }
}


function sortGaussiansByDepth(gaussians: GaussianScene, viewMatrix:Float32Array, sortingAlgorithm:string): Uint32Array {
    const calcDepth = (i:number) => gaussians.positions[i*3] * viewMatrix[2] +
                             gaussians.positions[i*3+1] * viewMatrix[6] +
                             gaussians.positions[i*3+2] * viewMatrix[10]

    const depthIndex = new Uint32Array(gaussians.count)

    // Default javascript sort [~0.9s]
    if (sortingAlgorithm == 'Array.sort') {
        const indices = new Array(gaussians.count)
            .fill(0)
            .map((_, i) => ({
                depth: calcDepth(i),
                index: i
            }))
            .sort((a, b) => a.depth - b.depth)
            .map(v => v.index)

        depthIndex.set(indices)
    }
    // Quick sort algorithm (Hoare partition scheme) [~0.4s]
    else if (sortingAlgorithm == 'quick sort') {
        const depths = new Float32Array(gaussians.count)

        for (let i = 0; i < gaussians.count; i++) {
            depthIndex[i] = i
            depths[i] = calcDepth(i)
        }

        quicksort(depths, depthIndex, 0, gaussians.count - 1)
    }
    // 16 bit single-pass counting sort [~0.3s]
    // https://github.com/antimatter15/splat
    else if (sortingAlgorithm == 'count sort') {
        let maxDepth = -Infinity;
        let minDepth = Infinity;
        let sizeList = new Int32Array(gaussians.count);

        for (let i = 0; i < gaussians.count; i++) {
            const depth = (calcDepth(i) * 4096) | 0

            sizeList[i] = depth
            maxDepth = Math.max(maxDepth, depth)
            minDepth = Math.min(minDepth, depth)
        }
        
        let depthInv = (256 * 256) / (maxDepth - minDepth);
        let counts0 = new Uint32Array(256*256);
        for (let i = 0; i < gaussians.count; i++) {
            sizeList[i] = ((sizeList[i] - minDepth) * depthInv) | 0;
            counts0[sizeList[i]]++;
        }
        let starts0 = new Uint32Array(256*256);
        for (let i = 1; i < 256*256; i++) starts0[i] = starts0[i - 1] + counts0[i - 1];
        for (let i = 0; i < gaussians.count; i++) depthIndex[starts0[sizeList[i]]++] = i;
    }

    return depthIndex;
}

// Quicksort algorithm - https://en.wikipedia.org/wiki/Quicksort#Hoare_partition_scheme
function quicksort(A:Float32Array, B:Float32Array|Uint32Array, lo:number, hi:number) {
    if (lo < hi) {
        const p = partition(A, B, lo, hi) 
        quicksort(A, B, lo, p)
        quicksort(A, B, p + 1, hi) 
    }
}

function partition(A:Float32Array, B: Float32Array|Uint32Array, lo:number, hi:number) {
    const pivot = A[Math.floor((hi - lo)/2) + lo]
    let i = lo - 1 
    let j = hi + 1
  
    while (true) {
        do { i++ } while (A[i] < pivot)
        do { j-- } while (A[j] > pivot)
    
        if (i >= j) return j
        
        let tmp = A[i]; A[i] = A[j]; A[j] = tmp // Swap A
            tmp = B[i]; B[i] = B[j]; B[j] = tmp // Swap B
    }    
}

export {}