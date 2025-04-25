import voxelproj

# Thin wrapper around voxelproj to mimic the tomosipo API


class VoxelOperator:
    def __init__(
        self,
        angles,
        z_order=2,
        additive=False,
        x_shape=None,
        y_shape=None,
    ):
        super(VoxelOperator, self).__init__()

        self.angles = angles
        self.z_order = z_order
        self.additive = additive
        self.x_shape = x_shape
        self.y_shape = y_shape
        self._transpose = BackprojectionVoxelOperator(self)

    def _fp(self, volume, out=None):
        if not self.additive and out is not None:
            out *= 0
        elif out is None:
            out = self.y_shape
        return voxelproj.forward(volume, self.angles, y=out, z_order=self.z_order)

    def _bp(self, projection, out=None):
        if not self.additive and out is not None:
            out *= 0
        elif out is None:
            out = self.x_shape
        return voxelproj.backward(projection, self.angles, x=out, z_order=self.z_order)

    def __call__(self, volume, out=None):
        return self._fp(volume, out)

    def transpose(self):
        return self._transpose

    @property
    def T(self):
        return self.transpose()


class BackprojectionVoxelOperator:
    def __init__(
        self,
        parent,
    ):
        super(BackprojectionVoxelOperator, self).__init__()
        self.parent = parent

    def __call__(self, projection, out=None):
        return self.parent._bp(projection, out)

    def transpose(self):
        return self.parent

    @property
    def T(self):
        return self.transpose()
