## 第14章 重载运算符与类型转换

### 14.1 基本概念

除了重载的函数调用运算符 operator() 之外，其它重载运算符不能含有默认实参。

当一个运算符函数是成员函数，this 绑定到左侧运算对象。

| 不能被重载的运算符 | 不建议被重载的运算符 |
| ------------------ | -------------------- |
| ::                 | &                    |
| .*                 | ,                    |
| .                  | &&                   |
| ?:                 | \|\|                 |

因为使用重载的运算符本质上是一次函数调用，所以关于运算对象求值顺序的规则无法应用到重载的运算符上。

**直接调用一个重载的运算符函数**

```c++
// 一个非成员函数的等价调用
data1 + data2;				// 普通的表达式
operator+(data1, data2);	// 等价的函数调用

data1 += data2;				// 基于“调用”的表达式
data1.operator+=(data2);	// 对成员运算符函数的等价调用
```

**一些建议**：

+ 如果定义 operator==，则也应该定义 operator!=
+ 如果类包含单序比较操作，则应该定义相关的全部关系运算符
+ 重载返回类型应该与内置返回类型相同（逻辑和关系运算返回 bool，算术运算返回类类型，赋值和复合赋值返回左侧运算对象引用）

**选择作为成员或者非成员**：

1. **成员**：=、[]、()、->、复合赋值运算、++、--、解引用
2. **非成员**：算术、相等性、关系、位运算（都是一些对称性运算符）

当定义为成员函数时，要求左侧运算对象必须是类对象，便不具备对称性了，故使用非成员函数。



### 14.2 输入和输出运算符

#### 14.2.1 重载输出运算符 <<

```c++
ostream& operator<<(ostream &os, const Sales_data &item) {
	// 输出运算符尽量减少格式化操作
	os << item.isbn() << " " << item.units_sold << " "
	   << item.revenue << " " << item.avg_price();
	return os;
}
```

重载输出运算符 << ，输入输出运算符必须是非成员函数

否则我们将需要同时修改标准库中的 ostream，istream 对象

同时一般声明为友元，方便读写类的非公有数据成员

#### 14.2.2 重载输入运算符

```c++
istream& operator>>(istream &is, Sales_data &item) {
	double price;
	is >> item.bookNo >> item.units_sold >> price;
	if (is)
		item.revenue = item.units_sold * price;
	else
		item = Sales_data();	// 输入失败：执行默认初始化
	return is;
}
```

输入运算符必须处理可能失败的情况，而输出运算符不需要。



### 14.3 算术和关系运算符

一般使用复合赋值来实现算术运算符

```c++
Sales_data operator+(const Sales_data &lhs, const Sales_data &rhs) {
	Sales_data sum = lhs;
	sum += rhs;
	return sum;
}

// 赋值运算符必须定义为类的成员
Sales_data&
Sales_data::operator+=(const Sales_data &rhs) {
	units_sold += rhs.units_sold;
	revenue += rhs.revenue;
	return *this;
}
```

##### 相等运算符和关系运算符

如果类定义了==，则这个类也应该定义!=

一般有 == 并不一定意味着存在一种逻辑可靠的 < 定义

当且仅当 < 的定义和 == 产生的结果一致时才定义 < 运算符

```c++
bool operator==(const Sales_data &lhs, const Sales_data &rhs) {
	return lhs.isbn() == rhs.isbn() &&
		lhs.units_sold == rhs.units_sold &&
		lhs.revenue == rhs.revenue;
}

// ==或!=应该把工作委托给另外一个
bool operator!=(const Sales_data &lhs, const Sales_data &rhs) {
	return !(lhs == rhs);
}
```



### 14.4 赋值运算符

```c++
class StrVec
{
public:
	StrVec& operator=(initializer_list<string>);		// 列表拷贝赋值
	// ...
};

vector<string> v;
v = { "a", "an", "the" };

// 赋值运算符重载必须定义为成员函数
StrVec&
StrVec::operator=(initializer_list<string> il) {
    // alloc_n_copy 分配内存空间并从给定范围内拷贝元素
	pair<string*, string*> data = alloc_n_copy(il.begin(), il.end());
	free();						// 销毁对象中的元素并释放空间
	elements = data.first;		// 更新数据成员使其指向新空间
	first_free = cap = data.second;
	return *this;
}
```



### 14.5 下标运算符

表示容器的类一般都会定义下标运算符 operator[]，同时下标运算符必须是成员函数。通常以所访问元素的引用作为返回值，这样的好处是下标可以出现在赋值运算符的任意一端。而且，一般会同时包含返回普通引用和常量引用的两个版本。
```c++
class StrVec
{
public:
	// 下标运算符必须是成员函数
	string& operator[](size_t n) { return elements[n]; }
	const string& operator[](size_t n) const { return elements[n]; }	// 返回常量引用，用于类的常量成员
    // ...
private:
	string* elements;		// 数组首元素
};
```



### 14.6 递增和递减运算符

递增和递减运算符存在前置和后置两种版本，一般两种版本都应该定义，而且应该被定义为类的成员函数。

**定义前置版本**

```c++
class StrBlobPtr
{
public:
	// 前置版本
	StrBlobPtr& operator++();
	StrBlobPtr& operator--();
	// ...
private:
	size_t curr;		// 数组当前位置
};

// 前置版本：返回引用
StrBlobPtr& StrBlobPtr::operator++() {
	check(curr, "increment past end of StrBlobPtr");
	++curr;
	return *this;
}
StrBlobPtr& StrBlobPtr::operator--() {
	--curr;
	check(curr, "decrement past begin of StrBlobPtr");
	return *this;
}
```

**定义后置版本**

为了区分前置和后置版本，后置版本接受一个额外的（不被使用）int 类型的形参，编译器为这个形参提供一个值为0的实参。

```c++
class StrBlobPtr
{
public:
    // 后置版本：返回原值
	StrBlobPtr operator++(int);
	StrBlobPtr operator--(int);
	// ...
};

StrBlobPtr StrBlobPtr::operator++(int) {	// 不使用，无需命名
	StrBlobPtr ret = *this;
	++*this;		// 后置运算符调用各自前置版本实现
	return ret;
}
StrBlobPtr StrBlobPtr::operator--(int) {
	StrBlobPtr ret = *this;
	--*this;
	return ret;
}

StrBlob a1 = { "hi", "bye", "now" };
StrBlobPtr p(a1);
p.operator++(0);	// 调用后置版本
p.operator++();		// 调用前置版本
```

后置运算符调用各自的前置版本来完成实际的工作



### 14.7 成员访问运算符

```c++
class StrBlobPtr
{
public:
	// 解引用运算符
	string& operator*() const {
		shared_ptr<vector<string>> p = check(curr, "dereference past end");
		return (*p)[curr];			// 返回引用
	}
	string* operator->() const {
		// 委托给解引用运算符
		return & this->operator*();	// 返回地址
	}
    // ...
};
```

箭头运算符必须是类的成员，解引用通常也是。

箭头运算符必须返回类的指针或者自定义了箭头运算符的某个类的对象



### 14.8 函数调用运算符

我们可以像使用函数一样使用重载了函数调用运算符的类的对象

```c++
struct absInt {
	int operator()(int val) const {
		return val < 0 ? -val : val;
	}
};

int i = -42;
absInt absObj;			// 含有函数调用运算符的对象，“函数对象”
int ui = absObj(i);		// 将 i 传递给 absObj.operator()
```

函数调用运算符必须是成员函数。一个类可以定义多个不同版本的调用运算符。

**含有状态的函数对象类**

定义一个打印 string 实参内容的类：

```c++
class PrintString
{
public:
	PrintString(ostream &o = cout, char c = ' ') :
		os(o), sep(c) {}
	void operator()(const string &s) const {
		os << s << sep;
	}
private:
	ostream &os;
	char sep;
};

string s = "hello";
PrintString printer;
printer(s);							// 使用默认方式打印 s
PrintString errors(cerr, '\n');		// 在 cerr 中打印 s，后跟一个换行符
errors(s);
```

函数对象常常作为泛型算法的实参，例如：

`for_each(vs.begin(), vs.end(), PrintString(cerr, '\n')/* 临时对象 */);`

这里的第三个实参是类型 PrintString 的一个临时对象。

#### 14.8.1 lambda 是函数对象

编译器会将 lambda 表达式翻译成一个未命名类的未命名对象，在 lambda 表达式产生的类中含有一个重载的函数调用运算符，例如：

```c++
vector<string> words = { "hi", "bye", "now" };
stable_sort(words.begin(), words.end(),
	[](const string &a, const string &b) {
		return a.size() < b.size();
	});

// 以上 lambda 表达式行为类似于下面这个类的一个未命名对象
class ShorterString {
public:
	bool operator()(const string &s1, const string &s2) const {
		return s1.size() < s2.size();
	}
};

// 重写后的 stable_sort
stable_sort(words.begin(), words.end(), ShorterString());
```

**表示 lambda 及相应捕获行为的类**

当 lambda 使用引用捕获变量时，程序会保证在执行时，引用的对象存在，而编译器可以直接使用，无需在 lambda 产生的类中存储为数据成员。

但是对于值捕获来说，通过值捕获的变量将被拷贝到 lambda 中，产生的类将会为每一个值捕获的变量建立对应的数据成员，并生成构造函数来初始化对应数据成员。

```c++
auto wc = find_if(words.begin(), words.end(),
	[sz](const string &a) {
		return a.size() >= sz;
	});

// 以上 lambda 表达式行为类似于下面这个类的一个未命名对象
class SizeComp {
public:
	SizeComp(size_t n) : sz(n) {}
	bool operator()(const string &s) const {
		return s.size() >= sz;
	}
private:
	size_t sz;
};

auto wc = find_if(words.begin(), words.end(), SizeComp(sz));
```

lambda 表达式产生的类不含有默认构造函数、赋值运算符、默认析构函数

#### 14.8.2 标准库定义的函数对象

标准库定义了一组表示算数运算符、关系运算符和逻辑运算符的类。

functional头文件中

| 算术          | 关系             | 逻辑           |
| ------------- | ---------------- | -------------- |
| plus<T>       | equal_to<T>      | logical_and<T> |
| minus<T>      | not_equal_to<T>  | logical_or<T>  |
| multiplies<T> | greater<T>       | logical_not<T> |
| divides<T>    | greater_equal<T> |                |
| modulus<T>    | less<T>          |                |
| negate<T>     | less_equal<T>    |                |

可以使用函数对象来实现两个指针比较地址大小

```c++
vector<string *> nameTable;
// 错误：直接使用默认的 < 将产生未定义行为
sort(nameTable.begin(), nameTable.end(),
    [](string *a, string *b) { return a < b; });
// 正确：标准库规定指针的 less 是定义良好的
sort(nameTable.begin(), nameTable.end(), less<string*>());
```

关联容器直接使用 less<key_type> 对元素排序，所以可以使用指针作为关联容器的关键值。

#### 14.8.3 可调用对象与function

**不同类型可能具有相同的调用形式**

```c++
// 普通函数
int add(int i. int j) { return i + j; }
// lambda 产生一个未命名的函数对象类
auto mod = [](int i, int j) { return i % j; };
// 函数对象类
struct divide {
    int operator()(int denominator, int divisor) {
        return denominator / divisor;
    }
}
```

以上这些可调用对象共享同一种调用形式：

`int(int, int)`

我们可以定义一个**函数表**用于存储指向这些可调用对象的“指针”

```c++
// 构建从运算符到函数指针的映射关系，其中函数接受两个 int、返回一个 int
map<string, int(*)(int, int)> binops;
// 正确：add是一个指向正确类型函数的指针
binpos.insert({"+", add});
binops.insert({"%", mod});		// 错误：mod不是一个函数指针
```

mod是一个 lambda 表达式，自动产生的类型与存储在 binops 中的类型不匹配。可以使用下面的 function 解决这个问题

##### 标准库 function 类型

定义在 functional 头文件中

| 名称                    | 解释                                                         |
| ----------------------- | ------------------------------------------------------------ |
| function<T> f;          | f是一个用来存储可调用对象的空function，<br />这些可调用对象的调用形式应该与函数类型T相同 |
| function<T> f(nullptr); | 显式地构造一个空function                                     |
| function<T> f(obj);     | 在f中存储可调用对象obj的副本                                 |
| f                       | 将f作为条件：当f含有一个可调用对象时为真；否则为假           |
| f(args)                 | 调用f中的对象，参数是args                                    |

定义为function<T>的成员的类型

| 名称                                                         | 解释                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| result_type                                                  | 该function类型的可调用对象返回的类型                         |
| argument_type<br />first_argument_type<br />second_argument_type | 当T有一个或两个实参时定义的类型。<br />如果T只有一个实参则，argument_type是该类型的同义词；<br />如果T有两个实参，则first_argument_type和second_argument_type分别代表两个实参的类型 |



```c++
/**
 * 所有可调用对象都必须接受两个int、返回一个int
 * 其中的元素可以是函数指针、函数对象或者 lambda
 */
map<string, function<int(int, int)>> binops = {
    {"+", add},						// 函数指针
    {"-", std::minus<int>()},		// 标准库函数对象
    {"/", divide()},				// 用户定义的函数对象
    {"*", [](int i, int j) { return i * j; }},	// 未命名的lambda
    {"%", mod}						// 命名了的lambda对象
};

binops["+"](10, 5);			// 调用 add(10, 5)
binops["-"](10, 5);			// 使用 minus<int> 对象的调用运算符
binops["/"](10, 5);			// 使用 divide 对象的调用运算符
binops["*"](10, 5);			// 调用 lambda 函数对象
binops["%"](10, 5);
```

**重载的函数与 function**

```c++
int add(int i, int j) { return i + j; }
Sales_data add(const Sales_data&, const Sales_data&);
map<string, function<int(int, int)>> binops;
```

以上会产生二义性问题，解决办法是存储函数指针而非函数名字：

```c++
int (*fp)(int, int) = add;			// 指针所指的 add 是接受两个 int 版本
binops.insert({"+", fp});

// 也可以使用 lambda 表达式来消除二义性
binops.insert({"+", [](int a, int b) { return add(a, b); }});
```

### 14.9 重载、类型转换与运算符

类类型转换：转换构造函数、类型转换运算符

#### 14.9.1 类型转换运算符

`operator type() const;`

不允许转换成数组或函数类型，但允许转换成指针或引用类型。

一般声明为类的成员函数，且不改变待转换对象的内容。没有返回类型，也没有形参。因为类型转换运算符是隐式执行的，所以无法传递实参。

```c++
class SmallInt {
public:
	SmallInt(int i = 0) : val(i) {	// 定义向类类型的转换
		if (i < 0 || i > 255)
			throw out_of_range("Bad SmallInt value");
	}
	operator int() const { return val; }	// 从类类型向其它类型转换
private:
	size_t val;
};

// 内置类型转换将 double 实参转换成 int
SmallInt si = 3.14;				// 调用 SmallInt(int) 构造函数
// SmallInt 的类型转换运算符将 si 转换成 int
si + 3.14;						// 内置类型转换将所得的 int 继续转换成 double
```

**注意**：避免过度使用类型转换函数，最适宜的情景是直接转换为对应的封装类，拥有一对一映射关系。

**类型转换运算符可能产生意外结果**

`int i = 42;`

`cin << i;`

程序错误地试图将输出运算符作用于输入流。

该代码会将 cin 转换为 bool，然后 bool 被提升称为 int 并作左移运算符左侧对象，最后提升后的 bool 值会被左移42个位置。

**显式类型转换运算符**

```c++
class SmallInt {
public:
	// ...
	explicit operator int() const { return val; }
	// ...
};

SmallInt si = 3;
si + 3;					// 错误：不能执行隐式转换
static_cast<int>(si) + 3;		// 正确：显式请求类型转换
```

例外情况：如果表达式被作为条件，显式的类型转换将被隐式地执行

所以一般将 operator bool 定义成 explicit 的。

#### 14.9.2 避免有二义性的类型转换

两种情况可能产生多重转换路径：

1. A类定义了一个接受B类对象的转换构造函数，B类定义了一个转换目标是A类的类型转换运算符。（只定义向一方的转换操作）
2. 类定义了多个转换规则。（只至多定义一个涉及算术类型的转换，剩下的交给标准类型转换完成）

**建议**：除了显式的向 bool 类型的转换之外，尽量避免定义类型转换函数，并限制隐式构造函数。

**实参匹配和相同的类型转换**

```c++
struct B;
struct A {
    A() = default;
    A(const B &);
    // ...
};
struct B {
    operator A() const;
    // ...
};
A f(const A &);				// 声明一个返回类型A，形参为常量A的函数
B b;
A a = f(b);					// 二义性错误：是f(B::operator A())，还是f(A::A(const B &))？

// 只能显式调用类型转换
A a1 = f(b.operator A());	// 正确：使用B的类型转换运算符
A a2 = f(A(b));				// 正确：使用A的构造函数
```

**二义性与转换目标为内置类型的多重类型转换**

例如：当两个转换函数的目标分别为 int 和 double 型（算术类型）时，若调用时需要转换到 long double类型，则两种类型转换都无法精确匹配，然而两种转换都可以使用，只需要后面再执行一次生成 long double 的标准类型转换即可，这时将产生二义性。当使用 long 初始化时也会遇到同样的问题。

但是把 short 提升成 int 优于把 short 转换成 double，因此这时不会产生二义性。

**重载函数与类型转换**：当涉及到多个重载函数时，将极有可能发生类型转换的二义性，建议直接使用构造函数显式调用。

#### 14.9.3 函数匹配与重载运算符

当我们使用表达式 a *sym* b 时需要考虑运算符的三种情况：成员函数版本、普通函数版本、类中的运算符重载版本

当我们调用普通命名函数时不会彼此重载，因为我们调用的形式不同。

当同时存在运算符重载和类型转换函数时可能产生二义性：

```c++
SmallInt s1, s2;
SmallInt s3 = s1 + s2;			// 使用重载的operator+
int i = s3 + 0;					// 二义性错误
```

第二条语句可以把0转换为 SmallInt，然后使用 SmallInt 的+，或者把 s3 转换为 int，执行内置加法运算。

**建议**：一般有运算符重载时，不要再使用类型转换函数，或者可以定义为显式的。

