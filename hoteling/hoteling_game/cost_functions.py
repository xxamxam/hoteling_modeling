class BaseRevenueFunction:
    def __init__(self, base_cost: float = 0.):
        self.base_cost = base_cost

    def __call__(self, distance: float) -> float:
        return self.base_cost


class FeeRevenueFunction(BaseRevenueFunction):
    def __init__(self, base_cost: float = 0, alpha: float = 1):
        super().__init__(base_cost)
        assert alpha > 0
        self.alpha = alpha

    def __call__(self, distance: float) -> float:
        return (max(0., self.base_cost - distance)) ** self.alpha

class AdditiveFeeRevenueFunction(BaseRevenueFunction):
    def __init__(self, base_cost: float = 0, alpha: float = 1):
        super().__init__(base_cost)
        self.alpha = alpha

    def __call__(self, distance: float) -> float:
        return self.base_cost - (distance + 1) ** self.alpha