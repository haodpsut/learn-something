from dwave.samplers import SimulatedAnnealingSampler
from dimod import BinaryQuadraticModel

# Định nghĩa BQM dưới dạng QUBO
# Hàm mục tiêu: x0*x1 - x0 - x1
linear = {'x0': -1, 'x1': -1}
quadratic = {('x0', 'x1'): 1}
bqm = BinaryQuadraticModel(linear, quadratic, 0.0, vartype='BINARY')

# Khởi tạo sampler
sampler = SimulatedAnnealingSampler()

# Chạy mô phỏng
sampleset = sampler.sample(bqm, num_reads=100)

# In kết quả
for sample, energy in sampleset.data(['sample', 'energy']):
    print(sample, "Energy:", energy)
