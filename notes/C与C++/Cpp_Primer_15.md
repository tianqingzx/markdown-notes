## 第15章 面向对象程序设计

### 15.1 OOP：概述

面向对象程序设计的核心思想

**数据抽象**：将类的接口与实现分离

**继承**：定义相似的类型并对相似关系建模

**动态绑定**：在一定程度上忽略相似类型的区别，而以统一的方式使用它们的对象。（当使用基类的引用或指针调用一个虚函数时将发生）

C++11新标准允许派生类显式地注明它将使用哪个成员函数改写基类的虚函数，即在形参列表之后增加一个 override 关键字。

### 15.2 定义基类和派生类

#### 15.2.1 定义基类

```c++
class Quote
{	// 基类：表示按原价销售的书籍
public:
	Quote() = default;		// 生成默认构造函数
	Quote(const string &book, double sales_price) :
		bookNo(book), price(sales_price) {}
	string isbn() const { return bookNo; }		// 返回书籍编号
	virtual double net_price(size_t n) const {	// 返回实际销售价格
		return n * price;
	}
	virtual ~Quote() = default;		// 一般根节点基类都会定义一个虚析构函数
private:
	string bookNo;
protected:
	double price = 0.0;
};
```

除了构造函数和静态函数都可以是虚函数，关键字 virtual 只能出现在类内部的声明语句之前而不能用于类外部的函数定义。同时在派生类中该函数隐式地也是虚函数。

成员函数如果没被声明为虚函数，则其解析过程发生在编译时而非运行时。

**访问控制与继承**：基类希望它的派生类有权访问其成员，同时禁止其他用户访问，就用**受保护的（protected）**访问运算符说明。

#### 15.2.2 定义派生类

派生类必须通过使用**类派生列表**明确指出从哪些基类继承而来。

```c++
class Bulk_quote : public Quote		// 类派生列表
{	// 表示可以打折销售的书籍
public:
	Bulk_quote() = default;
	Bulk_quote(const string &book, double p, size_t qty, double disc) :
		Quote(book, p)/* 调用基类构造函数初始化基类部分 */, min_qty(qty), discount(disc) {}
	double net_price(size_t) const override;	// 显式地注明将使用哪个成员改写基类的虚函数
private:
	size_t min_qty = 0;			// 适合折扣策略的最低购买量
	double discount = 0.0;		// 折扣额
};
```

**派生类对象及派生类向基类的类型转换**：

派生类对象中会含有其基类对应的组成部分，所以我们能够实现派生类到基类的类型转换。但是这种转换只能使用在将基类的指针或引用绑定到派生类对象中的基类部分上。

**派生类构造函数**

派生类只能使用基类的构造函数来初始化基类部分。（每个类控制它自己的成员初始化过程）

一般首先初始化基类的部分，然后再按照声明的顺序（不是按照初始化列表的顺序）依次初始化派生类成员。

**派生类使用基类的成员**

```c++
double Bulk_quote::net_price(size_t cnt) const {
	if (cnt >= min_qty)
		return cnt * (1 - discount) * price;
	else
		return cnt * price;
}
```

