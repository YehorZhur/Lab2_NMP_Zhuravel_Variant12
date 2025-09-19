import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt

def f(x):
    return 9*x**4 + 2*x**3 + 6*x**2 - 3*x

def find_segments():
    search_range = np.arange(-3, 3, 0.1)
    segments = []
    
    for i in range(len(search_range) - 1):
        x1, x2 = search_range[i], search_range[i + 1]
        if f(x1) * f(x2) < 0:  
            segments.append((round(x1, 1), round(x2, 1)))
    
    return segments

def bisection_method(a, b, eps=0.0001):
    print(f"\nМетод половинного ділення для відрізка [{a}, {b}]:")
    print("n\ta_n\t\tb_n\t\tx_n\t\tf(x_n)\t\t|b_n - a_n|")
    
    n = 0
    while abs(b - a) > eps:
        x_n = (a + b) / 2
        f_xn = f(x_n)
        
        print(f"{n}\t{a:.6f}\t{b:.6f}\t{x_n:.6f}\t{f_xn:.6f}\t{abs(b-a):.6f}")
        
        if f(a) * f_xn < 0:
            b = x_n
        else:
            a = x_n
        n += 1
    
    root = (a + b) / 2
    print(f"\nКорінь: x ≈ {root:.6f}")
    print(f"f({root:.6f}) = {f(root):.8f}")
    return root

def chord_method(a, b, eps=0.0001):
    print(f"\nМетод хорд для відрізка [{a}, {b}]:")
    
    f_second_derivative = nd.Derivative(f, n=2)

    if f(a) * f_second_derivative(a) > 0:
        x0 = a  
        xi = b
        print(f"Нерухомий кінець: a = {a}")
    else:
        x0 = b  
        xi = a
        print(f"Нерухомий кінець: b = {b}")
    
    print("n\tx_n\t\tf(x_n)\t\t|x_{n+1} - x_n|")
    
    n = 0
    while True:
        xi_1 = xi - (xi - x0) * f(xi) / (f(xi) - f(x0))
        
        print(f"{n}\t{xi:.6f}\t{f(xi):.8f}\t{abs(xi_1 - xi):.8f}")
        
        if abs(xi_1 - xi) < eps:
            break
            
        xi = xi_1
        n += 1
    
    print(f"\nКорінь: x ≈ {xi_1:.6f}")
    print(f"f({xi_1:.6f}) = {f(xi_1):.8f}")
    return xi_1

def analytical_root_separation():
    
    print("\nОчевидний корінь: x₁ = 0")
    
    def g(x):
        return 9*x**3 + 2*x**2 + 6*x - 3
    
    def g_prime(x):
        return 27*x**2 + 4*x + 6
    
    print("\nДля знаходження інших коренів розглядаємо:")
    print("g(x) = 9x³ + 2x² + 6x - 3 = 0")
    print("g'(x) = 27x² + 4x + 6")

    discriminant = 4**2 - 4*27*6
    print(f"Дискримінант g'(x): D = 16 - 648 = {discriminant}")
    print("Оскільки D < 0, g'(x) > 0 для всіх x, отже g(x) монотонно зростає")
    
    test_points = [-1, 0, 1]
    print("\nПеревірка знаків g(x):")
    for point in test_points:
        print(f"g({point}) = {g(point)}")
    
    print("\ng(-1) < 0, g(1) > 0, отже корінь g(x) знаходиться на проміжку (-1, 1)")
    
    return [(0, 0), (-1, 1)] 

print("Варіант 12: 9x⁴ + 2x³ + 6x² - 3x = 0")

intervals = analytical_root_separation()

segments = find_segments()
print("Знайдені проміжки зміни знаку функції:")
for seg in segments:
    print(f"[{seg[0]}, {seg[1]}]: f({seg[0]}) = {f(seg[0]):.4f}, f({seg[1]}) = {f(seg[1]):.4f}")

x = np.linspace(-2, 2, 1000)
y = [f(xi) for xi in x]

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='f(x) = 9x⁴ + 2x³ + 6x² - 3x')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Графік функції f(x) = 9x⁴ + 2x³ + 6x² - 3x')
plt.legend()
plt.show()

print("Корінь x₁ = 0 (точний)")

for a, b in segments:
    if a != b: 
        print(f"\n--- Уточнення кореня на проміжку [{a}, {b}] ---")

        root_bisection = bisection_method(a, b)
        

        root_chord = chord_method(a, b)
        
        print(f"\nПорівняння результатів:")
        print(f"Метод половинного ділення: x ≈ {root_bisection:.6f}")
        print(f"Метод хорд: x ≈ {root_chord:.6f}")
        print(f"Різниця: {abs(root_bisection - root_chord):.8f}")

print("Рівняння 9x⁴ + 2x³ + 6x² - 3x = 0 має корені:")
print("x₁ = 0.000000")
for a, b in segments:
    if a != b:
        root = bisection_method(a, b)
        print(f"x₂ ≈ {root:.6f}")
        break