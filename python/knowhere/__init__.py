from .swigknowhere import *
import numpy as np


def CreateIndex(index_name):
    if index_name == "annoy":
        return IndexAnnoy()
    if index_name == "ivf":
        return IVF()
    if index_name == "ivfsq":
        return IVFSQ()
    if index_name == "hnsw":
        return IndexHNSW()
    if index_name == "idmap":
        return IDMAP()
    if index_name == "binary_idmap":
        return BinaryIDMAP()
    if index_name == "gpu_ivf":
        return GPUIVF(-1)
    if index_name == "gpu_ivfpq":
        return GPUIVFPQ(-1)
    if index_name == "gpu_ivfsq":
        return GPUIVFSQ(-1)
    raise ValueError(
        """ index name only support 
            'annoy' 'ivf' 'ivfsq' 'hnsw' 'idmap' 'binary_idmap'
            'gpu_ivf', 'gpu_ivfsq', 'gpu_ivfpq'."""
    )


class GpuContext:
    def __init__(
        self, dev_id=0, pin_mem=200 * 1024 * 1024, temp_mem=300 * 1024 * 1024, res_num=2
    ):
        InitGpuResource(dev_id, pin_mem, temp_mem, res_num)

    def __del__(self):
        ReleaseGpuResource()


def UnpackRangeResults(results, nq):
    lims = np.zeros(
        [
            nq + 1,
        ],
        dtype=np.int32,
    )
    DumpRangeResultLimits(results, lims)
    dis = np.zeros(
        [
            lims[-1],
        ],
        dtype=np.float32,
    )
    DumpRangeResultDis(results, dis)
    ids = np.zeros(
        [
            lims[-1],
        ],
        dtype=np.int32,
    )
    DumpRangeResultIds(results, ids)

    dis_list = []
    ids_list = []

    for idx in range(nq):
        dis_list.append(dis[lims[idx] : lims[idx + 1]])
        ids_list.append(ids[lims[idx] : lims[idx + 1]])

    return ids_list, dis_list
