### 5.1 Singleton 模式

程序在运行时，通常会生成很多的实例。例如，表示字符串的`java.lang.String`类的实例与字符串是一对一关系 所以当有1000个字符串的时候，会生成1000个实例。

但是有时候又需要：

+ **想确保任何情况下都绝对只有1个实例**

+ **想在程序上表现出“只存在一个实例”**

像这样的确保只存在一个实例的模式被称为*Singleton*模式。*Singleton*是指只含有一个元素的集合。因为本模式只能生成一个实例，因此以*Singleton*命名。

### 5.2 示例程序

**类的一览表**

| 名字      | 说明               |
| --------- | ------------------ |
| Singleton | 只存在一个实例的类 |
| Main      | 测试程序行为的类   |

**示例程序的类图**

<img src="F:\文档\Typora Files\markdown-notes\images\notes\设计模式\singleton_uml.png" alt="singleton_uml" style="zoom:80%;" />

构造函数*Singleton*前带有“-”，表示*Singleton*函数是*private*。此外，*getInstance*方法带有下划线，表示该方法是*static*方法。

> Singleton 类

*Singleton*类只会生成一个实例。*Singleton*类定义了*static*字段（类成员变量）*singleton*，并将其初始化为*Singleton*类的实例。初始化行为仅仅在该类被加载时进行一次。

*Singleton*类的**构造函数是*private***的，这是为了禁止从*Singleton*类外部调用构造函数。

```java
public class Singleton {
    private static Singleton singleton = new Singleton();
    private Singleton() {
        System.out.println("生成了一个实例。");
    }
    public static Singleton getInstance() {
        return singleton;
    }
}
```

> Main 类

```java
public class Main {
    public static void main(String[] args) {
        System.out.println("Start.");
        Singleton obj1 = Singleton.getInstance();
        Singleton obj2 = Singleton.getInstance();
        if (obj1 == obj2) {
            System.out.println("obj1 与 obj2 是相同的实例。");
        } else {
            System.out.println("obj1 与 obj2 是不同的实例。");
        }
        System.out.println("End.");
    }
}
```

**运行结果**

<img src="F:\文档\Typora Files\markdown-notes\images\notes\设计模式\singleton.png" alt="singleton" style="zoom:80%;" />

### 5.3 Singleton 模式中的登场角色

+ **Singleton**

在*Singleton*模式中，只有*Singleton*这一个角色。*Singleton*角色中有一个返回唯一实例的*static*方法。该方法总会返回一个实例。

### 5.4 相关设计模式

+ AbstractFactory模式
+ Builder模式
+ Facade模式
+ Prototype模式