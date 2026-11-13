#include <iostream>

int countOnesInFirstFiveBits(int num)
{
    int count = 0;

    for (int i = 0; i < 5; i++)
    {
        if (num & (1 << i))
        {
            count++;
        }
    }

    return count;
}

int main()
{
    int num = 268452398; // 假设这是一个8位的二进制数
    int num_tem = num>>(2*2);
    // int onesCount = countOnesInFirstFiveBits(num);
    int mask = (1 << 0) -1;
    int res = num & mask;
    std::cout << "前32位二进制数中1的个数为: " << num_tem << std::endl;
        std::cout << "前32位二进制数中1的个数为: " << (num_tem&1) << std::endl;
            std::cout << "前32位二进制数中1的个数为: " << ((num_tem>>1)&1) << std::endl;
    // 获取第6位-t5的值
    int fifthBit = (num >> 5) & 1;

    std::cout << "The fifth bit of " << num << " in binary is: " << fifthBit << std::endl;

    return 0;
}
