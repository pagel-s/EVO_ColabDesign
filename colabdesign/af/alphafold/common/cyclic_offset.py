import numpy as np


def add_cyclic_offset(self, bug_fix=True):
    '''add cyclic offset to connect N and C term'''
    def cyclic_offset(L):
        i = np.arange(L)
        ij = np.stack([i,i+L],-1)
        offset = i[:,None] - i[None,:]
        c_offset = np.abs(ij[:,None,:,None] - ij[None,:,None,:]).min((2,3))
        if bug_fix:
            a = c_offset < np.abs(offset)
            c_offset[a] = -c_offset[a]
        return c_offset * np.sign(offset)
    idx = self._inputs["residue_index"]
    offset = np.array(idx[:,None] - idx[None,:])

    if self.protocol == "binder":
        c_offset = cyclic_offset(self._binder_len)
        offset[self._target_len:,self._target_len:] = c_offset
    
    if self.protocol in ["fixbb","partial","hallucination"]:
        Ln = 0
        for L in self._lengths:
            offset[Ln:Ln+L,Ln:Ln+L] = cyclic_offset(L)
        Ln += L
    self._inputs["offset"] = offset
