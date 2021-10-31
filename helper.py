import numpy
from sympy import symbols, Eq, solve, log

def partial(element, function):
	"""
	partial : sympy.core.symbol.Symbol * sympy.core.add.Add -> sympy.core.add.Add
	partial(element, function) Performs partial derivative of a function of several variables is its derivative with respect to one of those variables, with the others held constant. Return partial_diff.
	"""
	partial_diff = function.diff(element)

	return partial_diff


def gradient(partials):
	"""
	gradient : List[sympy.core.add.Add] -> numpy.matrix
	gradient(partials) Transforms a list of sympy objects into a numpy matrix. Return grad.
	"""
	grad = numpy.matrix([[partials[0]], [partials[1]]])

	return grad


def gradient_to_zero(symbols_list, partials):
	"""
	gradient_to_zero : List[sympy.core.symbol.Symbol] * List[sympy.core.add.Add] -> Dict[sympy.core.numbers.Float]
	gradient_to_zero(symbols_list, partials) Solve the null equation for each variable, and determine the pair of coordinates of the singular point. Return singular.
	"""
	partial_x = Eq(partials[0], 0)
	partial_y = Eq(partials[1], 0)

	singular = solve((partial_x, partial_y), (symbols_list[0], symbols_list[1]))

	return singular

def hessian(partials_second, cross_derivatives):
	"""
	hessian : List[sympy.core.add.Add] * sympy.core.add.Add -> numpy.matrix
	hessian(partials_second, cross_derivatives) Transforms a list of sympy objects into a numpy hessian matrix. Return hessianmat.
	"""
	hessianmat = numpy.matrix([[partials_second[0], cross_derivatives], [cross_derivatives, partials_second[1]]])

	return hessianmat


def determat(partials_second, cross_derivatives, singular, symbols_list):
	"""
	List[sympy.core.add.Add] * sympy.core.add.Add * Dict[sympy.core.numbers.Float] * List[sympy.core.symbol.Symbol] -> sympy.core.numbers.Float
	determat(partials_second, cross_derivatives, singular, symbols_list) Computes the determinant of the Hessian matrix at the singular point. Return det.
	"""
	det = partials_second[0].subs([(symbols_list[0], singular[symbols_list[0]]), (symbols_list[1], singular[symbols_list[1]])]) * partials_second[1].subs([(symbols_list[0], singular[symbols_list[0]]), (symbols_list[1], singular[symbols_list[1]])]) - (cross_derivatives.subs([(symbols_list[0], singular[symbols_list[0]]), (symbols_list[1], singular[symbols_list[1]])]))**2

	return det
  
def main():
	"""
	Fonction principale.
	"""
	x, y = symbols('x y')
	symbols_list = [x, y]
	function = x**2 - (3/2)*x*y + y**2
	partials, partials_second = [], []

	for element in symbols_list:
		partial_diff = partial(element, function)
		partials.append(partial_diff)

	grad = gradient(partials)
	singular = gradient_to_zero(symbols_list, partials)

	cross_derivatives = partial(symbols_list[0], partials[1])

	for i in range(0, len(symbols_list)):
		partial_diff = partial(symbols_list[i], partials[i])
		partials_second.append(partial_diff)

	hessianmat = hessian(partials_second, cross_derivatives)
	det = determat(partials_second, cross_derivatives, singular, symbols_list)

	print("Hessian matrix that organizes all the second partial derivatives of the function {0} is :\n {1}".format(function, hessianmat))
	print("Determinant in the singular point {0} is :\n {1}".format(singular, det))

main()