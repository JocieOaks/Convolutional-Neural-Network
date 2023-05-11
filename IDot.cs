// See https://aka.ms/new-console-template for more information
public interface IDot<T>
{
    T Multiply(T other);
    T Multiply(float multiple);
    T Random();
    T Divide(T other);
    T ReLU();
    T Add(T other);
    T Add(float value);
    T Subtract(T other);
    T Pow(float power);
}
