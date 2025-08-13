import numpy

# male: avg 69.1  stdev 2.9
# female: avg 63.7 stdev 2.7


male_sample = numpy.random.normal(69.1,2.9, 1000)
female_sample = numpy.random.normal(63.7, 2.7, 1000)

print(male_sample)
print(female_sample)

male_minimum_height=69.1 + 2*2.9
female_minimum_height = 63.7 + 2 * 2.7
print(f"male taller than {male_minimum_height} and female taller than {female_minimum_height}")