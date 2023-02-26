### iota

Fills the range `[first, last)` with sequentially increasing values, starting with `value` and repetitively evaluating ++value.

```c++
template<class ForwardIt, class T>
constexpr // since C++20
void iota(ForwardIt first, ForwardIt last, T value)
{
    while(first != last) {
        *first++ = value;
        ++value;
    }
}
```

### accumulate

Computes the sum of the given value `init` and the elements in the range `[first, last)`.

![image-20220920183124702](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20220920183124702.png)

### 结构体构造函数

```c++
#include <iostream>
#include <string>
using namespace std;
struct node{
	int data;
	string str;
	char x;
	//自己写的初始化函数
	void init(int a, string b, char c){
		this->data = a;
		this->str = b;
		this->x = c;
	}
	node() :x(), str(), data(){}
	node(int a, string b, char c) :x(c), str(b), data(a){}
}N[10];
int main()
{
	  N[0] = { 1,"hello",'c' };  
	  N[1] = { 2,"c++",'d' };    //无参默认结构体构造体函数
	  N[2].init(3, "java", 'e'); //自定义初始化函数的调用
	  N[3] = node(4, "python", 'f'); //有参数结构体构造函数
	  N[4] = { 5,"python3",'p' };

	//现在我们开始打印观察是否已经存入
	for (int i = 0; i < 5; i++){
		cout << N[i].data << " " << N[i].str << " " << N[i].x << endl;
	}
	system("pause");
	return 0;
}
```

