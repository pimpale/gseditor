import { mat4 } from "gl-matrix"
import assert from "../../utils/assert"
import { GaussianObjectInput } from "./sceneLoader"


type GaussianContextualizedObject = {
    id: number
    obj: GaussianObjectInput,
}

type SortWorkerInput = {
    viewMatrix: Float32Array
    sortingAlgorithm: string
    sceneGraph: Map<number, { transform: mat4, object: GaussianObjectInput }>
}

// a processed input object with object ids
export type ProcessedGaussianScene = {
    count: number,
    sceneMin: Float32Array,
    sceneMax: Float32Array,
    positions: Float32Array,
    opacities: Float32Array,
    colors: Float32Array,
    cov3Da: Float32Array,
    cov3Db: Float32Array,
    objectIds: Uint32Array
}

onmessage = (event: MessageEvent<SortWorkerInput>) => {

    console.log("[Worker] Received message")

    const { sceneGraph, viewMatrix, sortingAlgorithm } = event.data
    
    const start = performance.now()

    const count = [...sceneGraph.values()].reduce((a, b) => a + b.object.count, 0)

    // create a merged object for fast lookup
    const gaussians: ProcessedGaussianScene = {
        count,
        sceneMin: Float32Array.from([Infinity, Infinity, Infinity]),
        sceneMax: Float32Array.from([-Infinity, -Infinity, -Infinity]),
        colors: new Float32Array(count * 3),
        positions: new Float32Array(count * 3),
        opacities: new Float32Array(count),
        cov3Da: new Float32Array(count * 3),
        cov3Db: new Float32Array(count * 3),
        objectIds: new Uint32Array(count),
    }

    let offset = 0;
    for (const [key, value] of sceneGraph.entries()) {
        const id = key
        const g = value.object;
        gaussians.sceneMin = gaussians.sceneMin.map((v, j) => Math.min(v, g.sceneMin[j]))
        gaussians.sceneMax = gaussians.sceneMax.map((v, j) => Math.max(v, g.sceneMax[j]))
        gaussians.colors.set(g.colors, offset * 3)
        gaussians.positions.set(g.positions, offset * 3)
        gaussians.opacities.set(g.opacities, offset)
        gaussians.cov3Da.set(g.cov3Da, offset * 3)
        gaussians.cov3Db.set(g.cov3Db, offset * 3)
        gaussians.objectIds.fill(id, offset, offset + g.count)
        offset += g.count
    }

    // create an empty object to write to
    const data: ProcessedGaussianScene = {
        count,
        sceneMin: gaussians.sceneMin,
        sceneMax: gaussians.sceneMax,
        colors: new Float32Array(count * 3),
        positions: new Float32Array(count * 3),
        opacities: new Float32Array(count),
        cov3Da: new Float32Array(count * 3),
        cov3Db: new Float32Array(count * 3),
        objectIds: new Uint32Array(count),
    }

    // Sort the gaussians!
    const depthIndex = sortGaussiansByDepth(gaussians.positions, viewMatrix, sortingAlgorithm)

    // Update arrays containing the data
    for (let j = 0; j < depthIndex.length; j++) {
        const i = depthIndex[j]

        data.colors[j * 3] = gaussians.colors[i * 3]
        data.colors[j * 3 + 1] = gaussians.colors[i * 3 + 1]
        data.colors[j * 3 + 2] = gaussians.colors[i * 3 + 2]

        data.positions[j * 3] = gaussians.positions[i * 3]
        data.positions[j * 3 + 1] = gaussians.positions[i * 3 + 1]
        data.positions[j * 3 + 2] = gaussians.positions[i * 3 + 2]

        data.opacities[j] = gaussians.opacities[i]

        // Split the covariance matrix into two vec3
        // so they can be used as vertex shader attributes
        data.cov3Da[j * 3] = gaussians.cov3Da[i * 3]
        data.cov3Da[j * 3 + 1] = gaussians.cov3Da[i * 3 + 1]
        data.cov3Da[j * 3 + 2] = gaussians.cov3Da[i * 3 + 2]

        data.cov3Db[j * 3] = gaussians.cov3Db[i * 3]
        data.cov3Db[j * 3 + 1] = gaussians.cov3Db[i * 3 + 1]
        data.cov3Db[j * 3 + 2] = gaussians.cov3Db[i * 3 + 2]
    }

    // Check that the arrays are the correct length
    assert(data.cov3Da.length == 3 * gaussians.count, "cov3Da count mismatch")
    assert(data.cov3Db.length == 3 * gaussians.count, "cov3Db count mismatch")

    const sortTime = `${((performance.now() - start) / 1000).toFixed(3)}s`
    console.log(`[Worker] Sorted ${gaussians.count} gaussians in ${sortTime}. Algorithm: ${sortingAlgorithm}`)

    postMessage({
        data, sortTime,
    }, [
        data.positions.buffer,
        data.colors.buffer,
        data.opacities.buffer,
        data.cov3Da.buffer,
        data.cov3Db.buffer,
        data.objectIds.buffer,
    ])

}



function sortGaussiansByDepth(gaussian_positions: Float32Array, viewMatrix: Float32Array, sortingAlgorithm: string): Uint32Array {
    const n_gaussians = gaussian_positions.length / 3

    const calcDepth = (i: number) => gaussian_positions[i * 3] * viewMatrix[2] +
        gaussian_positions[i * 3 + 1] * viewMatrix[6] +
        gaussian_positions[i * 3 + 2] * viewMatrix[10]

    const depthIndex = new Uint32Array(n_gaussians)

    // Default javascript sort [~0.9s]
    if (sortingAlgorithm == 'Array.sort') {
        const indices = new Array(n_gaussians)
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
        const depths = new Float32Array(n_gaussians)

        for (let i = 0; i < n_gaussians; i++) {
            depthIndex[i] = i
            depths[i] = calcDepth(i)
        }

        quicksort(depths, depthIndex, 0, n_gaussians - 1)
    }
    // 16 bit single-pass counting sort [~0.3s]
    // https://github.com/antimatter15/splat
    else if (sortingAlgorithm == 'count sort') {
        let maxDepth = -Infinity;
        let minDepth = Infinity;
        let sizeList = new Int32Array(n_gaussians);

        for (let i = 0; i < n_gaussians; i++) {
            const depth = (calcDepth(i) * 4096) | 0

            sizeList[i] = depth
            maxDepth = Math.max(maxDepth, depth)
            minDepth = Math.min(minDepth, depth)
        }

        let depthInv = (256 * 256) / (maxDepth - minDepth);
        let counts0 = new Uint32Array(256 * 256);
        for (let i = 0; i < n_gaussians; i++) {
            sizeList[i] = ((sizeList[i] - minDepth) * depthInv) | 0;
            counts0[sizeList[i]]++;
        }
        let starts0 = new Uint32Array(256 * 256);
        for (let i = 1; i < 256 * 256; i++) starts0[i] = starts0[i - 1] + counts0[i - 1];
        for (let i = 0; i < n_gaussians; i++) depthIndex[starts0[sizeList[i]]++] = i;
    }

    return depthIndex;
}

// Quicksort algorithm - https://en.wikipedia.org/wiki/Quicksort#Hoare_partition_scheme
function quicksort(A: Float32Array, B: Float32Array | Uint32Array, lo: number, hi: number) {
    if (lo < hi) {
        const p = partition(A, B, lo, hi)
        quicksort(A, B, lo, p)
        quicksort(A, B, p + 1, hi)
    }
}

function partition(A: Float32Array, B: Float32Array | Uint32Array, lo: number, hi: number) {
    const pivot = A[Math.floor((hi - lo) / 2) + lo]
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