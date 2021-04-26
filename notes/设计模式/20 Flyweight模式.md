# 第20章 Flyweight模式

### 共享对象，避免浪费



### 20.1 Flyweight模式

*Flyweight*是“轻量级”的意思，指的是拳击比赛中选手体重最轻的等级。顾名思义，该设计模式的作用是为了让对象变“轻”。

对象在计算机中是虚拟存在的东西，它的“重”和“轻”并非指实际重量，而是它们“所使用的内存大小”。使用内存多的对象就是“重”对象，使用内存少的对象就是“轻”对象。

在*Java*中，可以通过以下语句生成*Something*类的实例。

`new Something()`

为了能够在计算机中保存对象，需要分配给其足够的内存空间。当程序中需要大量对象时，如果都使用了*new*关键字来分配内存，将会消耗大量内存空间。

关于*Flyweight*模式，一言以蔽之就是“**通过尽量共享实例来避免*new*出实例**”。

当需要某个实例时，并不总是通过*new*关键字来生成实例，而是尽量共用已经存在的实例。这就是*Flyweight*模式的核心内容。



### 20.2 示例程序

在示例程序中，有一个将许多普通字符组合成为“大型字符”的类，它的实例就是重实例。为了进行测试，我们以文件形式保存了大型字符‘0’～‘9’和‘-’的字体数据。

**类的一览表**

| 名字           | 说明                                  |
| -------------- | ------------------------------------- |
| BigChar        | 表示“大型字符”的类                    |
| BigCharFactory | 表示生成和共用BigChar类的实例的类     |
| BigString      | 表示多个BigChar组成的“大型字符串”的类 |
| Main           | 测试程序行为的类                      |

*BigChar*是表示“大型字符”的类。它会从文件中读取大型字符的字体数据，并将它们保存在内容中，然后使用*print*方法输出大型字符。大型字符会消耗很多内存，因此我们需要考虑如何共享*BigChar*类的实例。

*BigCharFactory*类会根据需要生成*BigChar*类的实例。不过如果它发现之前已经生成了某个大型字符的*BigChar*类的实例，则会直接利用该实例，而不会再生成新的实例。生成的实例全部被保存在*pool*字段中。此外，为了能够快速查找出之前是否已经生成了某个大型字符所对应的实例，我们使用了*java.util.Hashmap*类。

*BigString*类用于将多个*BigChar*组成“大型字符串”。

*Main*类是用于测试程序行为的类。

**示例程序的类图**

![flyweight_uml](F:\文档\Typora Files\markdown-notes\images\notes\设计模式\flyweight_uml.PNG)

> BigChar类

*BigChar*类是表示“大型字符”的类。

它的构造函数会生成接收到的字符所对应的“大型字符”版本的实例，并将其保存在*fontdata*字段中。例如，如果构造函数接收到的字符是‘3’，那么在fontdata字段中保存的就是下面这样的字符串（为了方便，我们在*”\n”*后换行了）。

。。。

我们将组成这些“大型字符”的数据（即字体数据）保存在文件中。文件的命名规则是在该字体数据所代表的字符前加上*”big”*，文件后缀名是*”.txt”*。例如，*‘3’*对应的字体数据保存在*”big3.txt”*文件中。如果找不到某个字符对应的字体数据，就在该字符后面打上问号（*“？”*）作为其字体数据。

在该类中，没有出现关于*Flyweight*模式中“共享”的相关代码。关于控制共享的代码，请看*BigCharFactory*类。

```java
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class BigChar {
    // 字符名字
    private char charname;
    // 大型字符对应的字符串（由'#' '.' '\n'组成）
    private String fontdata;
    // 构造函数
    public BigChar(char charname) {
        this.charname = charname;
        try {
            BufferedReader reader = new BufferedReader(
                    new FileReader("big" + charname + ".txt")
            );
            String line;
            StringBuffer buf = new StringBuffer();
            while ((line = reader.readLine()) != null) {
                buf.append(line);
                buf.append("\n");
            }
            reader.close();
            this.fontdata = buf.toString();
        } catch (IOException e) {
            this.fontdata = charname + "?";
        }
    }
    // 显示大型字符
    public void print() {
        System.out.println(fontdata);
    }
}
```



> BigCharFactory类

*BigCharFactory*类是生成*BigChar*类的实例的工厂（*factory*）。它实现了共享实例的功能。

*pool*字段用于管理已经生成的*BigChar*类的实例。*Pool*有泳池的意思。现在任何存放某些东西的地方都可以被叫作*Pool*。泳池存储的是水，而*BigCharFactory*的*pool*中存储的则是已经生成的*BigChar*类的实例。

在*BigCharFactory*类中，我们使用*java.util.HashMap*类来管理“字符串->实例”之间的对应关系。使用*java.util.HashMap*类的*put*方法可以将某个字符串（键）与一个实例（值）关联起来。之后，就可以通过键来获取它相应的值。在示例程序中，我们将接收到的单个字符（例如‘3’）作为键与表示3的*BigChar*的类的实例对应起来。

我们使用了*Singleton*模式来实现*BigCharFactory*类，这是因为我们只需要一个*BigCharFactory*类的实例就可以了。*getInstance*方法用于获取*BigCharFactory*类的实例（注意不是*BigChar*类的实例）。

*getBigChar*方法是*Flyweight*模式的核心方法。该方法会生成接收到的字符所对应的*BigChar*类的实例。不过，如果它发现字符所对应的实例已经存在，就不会再生成新的实例，而是将之前的那个实例返回给调用者。

该方法首先会通过*pool.get()*方法查找，以调查是否存在接收到的字符（*charname*）所对应的*BigChar*类的实例。如果返回值为*null*，表示目前为止还没有创建该实例，于是它会通过*new BigChar(charname);*来生成实例，并通过*pool.put*将该实例放入*HashMap*中。如果返回值不为*null*，则会将之前生成的实例返回给调用者。

为什么我们要使用*synchronized*关键字修饰*getBigChar*方法呢？

```java
import java.util.HashMap;

public class BigCharFactory {
    // 管理已经生成的 BigChar 的实例
    private HashMap pool = new HashMap();
    // Singleton 模式
    private static BigCharFactory singleton = new BigCharFactory();
    // 构造函数
    private BigCharFactory() {}
    // 获取唯一的实例
    public static BigCharFactory getInstance() {
        return singleton;
    }
    // 生成（共享）BigChar类的实例
    public synchronized BigChar getBigChar(char charname) {
        BigChar bc = (BigChar) pool.get("" + charname);
        if (bc == null) {
            bc = new BigChar(charname);     // 生成BigChar的实例
            pool.put("" + charname, bc);
        }
        return bc;
    }
}
```



> BigString类

*BigString*类表示由*BigChar*组成的“大型字符串”的类。

*bigchars*字段是*BigChar*类型的数组，它里面保存着*BigChar*类的实例。在构造函数的*for*语句中，我们并没有像下面这样使用*new*关键字来生成*BigChar*类的实例。

```java
for (int i = 0; i < bigchars.length; i++) {
    bigchars[i] = new BigChar(string.charAt(i));<-不共享实例
}
```

而是调用了*getBigChar*方法，具体如下。

```java
BigCharFactory factory = BigCharFactory.getInstance();
for (int i = 0; i < bigchars.length; i++) {
    bigchars[i] = factory.getBigChar(string.charAt(i));<-共享实例
}
```

由于调用了*BigCharFactory*方法，所以对于相同的字符来说，可以实现*BigChar*类的实例共享。例如，当要生成字符串*“1212123”*对应的*BigString*类的实例时。

```java
public class BigString {
    // “大型字符”的数组
    private BigChar[] bigchars;
    // 构造函数
    public BigString(String string) {
        bigchars = new BigChar[string.length()];
        BigCharFactory factory = BigCharFactory.getInstance();
        for (int i = 0; i < bigchars.length; i++) {
            bigchars[i] = factory.getBigChar(string.charAt(i));
        }
    }
    // 显示
    public void print() {
        for (int i = 0; i < bigchars.length; i++) {
            bigchars[i].print();
        }
    }
}
```



> Main类

*Main*类比较简单。它根据接收到的参数生成并显示*BigString*类的实例。

```java
public class Main {
    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("Usage: java Main digits");
            System.out.println("Example: java Main 1212123");
            System.exit(0);
        }

        BigString bs = new BigString(args[0]);
        bs.print();
    }
}
```



### 20.3 Flyweight模式中的登场角色

在*Flyweight*模式中有以下登场角色。类图如下：

+ ***Flyweight*（轻量级）**

按照通常方式编写程序会导致程序变重，所以如果能够共享实例会比较好，而*Flyweight*角色表示的就是那些实例会被共享的类。在示例程序中，由*BigChar*类扮演此角色。

+ ***FlyweightFactory*（轻量级工厂）**

*FlyweightFactory*角色是生成*Flyweight*角色的工厂。在工厂中生成*Flyweight*角色可以实现共享实例。在示例程序中，由*BigCharFactory*类扮演此角色。

+ ***Client*（请求者）**

*Client*角色使用*FlyweightFactory*角色来生成*Flyweight*角色。在示例程序中，由*BigString*类扮演此角色。



### 20.4 拓展思路的要点

> 对多个地方产生影响

*Flyweight*模式的主题是“共享”。那么，在共享实例时应当注意什么呢？

首先要想到的是“如果要改变被共享的对象，就会对多个地方产生影响”。也就是说，一个实例的改变会同时反映到所有使用该实例的地方。例如，假设我们改变了示例程序中*BigChar*类的‘3’所对应的字体数据，那么*BigString*类中使用的所有‘3’的字体（形状）都会发生改变。在编程时，像这样修改一个地方会对多个地方产生影响并非总是不好。有些情况下这是好事，有些情况下这是坏事。不管怎样，“修改一个地方会对多个地方产生影响”，这就是共享的特点。

因此，在决定*Flyweight*角色中的字段时，需要精挑细选。只将那些真正应该在多个地方共享的字段定义在*Flyweight*角色中即可。

关于这一点，让我们简单地举个例子。假设我们要在示例程序中增加一个功能，实现显示“带颜色的大型文字”。那么此时，颜色信息应当放在哪个类中呢？

首先，假设我们将颜色信息放在*BigChar*类中。由于*BigChar*类的实例是被共享的，因此颜色信息也被共享了。也就是说，*BigString*类中用到的所有*BigChar*类的实例都带有相同的颜色。

如果我们不把颜色信息放在*BigChar*类中，而是将它放在*BigString*类中。那么*BigString*类会负责管理“第三个字符的颜色是红色的”这样的颜色信息。这样一来，我们就可以实现以不同的颜色显示同一个*BigChar*类的实例。

> Intrinsic与Extrinsic

应当共享的信息被称作*Intrinsic*信息。*Intrinsic*的意思是“本质的”“固有的”。换言之，它指的是不论实例在哪里，不论在什么情况下都不会改变的信息，或是不依赖于实例状态的信息。在示例程序中，*BigChar*的字体数据不论在*BigString*中的哪个地方都不会改变。因此，*BigChar*的字体数据属于*Intrinsic*信息。

另一方面，不应当共享的信息被称作*Extrinsic*信息。*Extrinsic*的意思是“外在的”“非本质的”。也就是说，它是当实例的位置、状况发生改变时会变化的信息，或是依赖于实例状态的信息。在示例程序中，*BigChar*的实例在*BigString*中是第几个字符这种信息会根据*BigChar*在*BigString*中的位置变化而发生变化。因此，不应当在*BigChar*中保存这个信息，它属于*Extrinsic*信息。

因此，前面提到的是否共享“颜色”信息这个问题，我们也可以换种说法，即应当将“颜色”看作是*Intrinsic*信息还是*Extrinsic*信息。

**Intrinsic与Extrinsic信息**

| Intrinsic信息                | Extrinsic信息              |
| ---------------------------- | -------------------------- |
| 不依赖于位置与状况，可以共享 | 依赖于位置与状况，不能共享 |

> 不要让被共享的实例被垃圾回收器回收了

在*BigCharFactory*类中，我们使用*java.util.HashMap*来管理已经生成的*BigChar*的实例。像这样在*Java*中自己“管理”实例时，必须注意“不要让实例被垃圾回收器回收了”。

下面我们简单地学习一下*Java*中的垃圾回收器。在*Java*程序中可以通过*new*关键字分配内存空间。如果分配了过多内存，就会导致内存不足。这时，运行*Java*程序的虚拟机就会开始**垃圾回收处理**。它会查看自己的内存空间（堆空间）中是否存在没有被使用的实例，如果存在就释放该实例，这样就可以回收可用的内存空间。总之，它像是垃圾回收车一样回收那些不再被使用的内存空间。

得益于垃圾回收器，*Java*开发人员对于*new*出来的实例可以放任不管（在*C++*中，使用*new*关键字分配内存空间后，必须显式地使用*delete*关键字释放内存空间。不过在*Java*中没有必要进行*delete*处理。当然，*Java*也没有提供*delete*关键字）。

此处的关键是垃圾回收器会“释放没有被使用的实例”。垃圾回收器在进行垃圾回收的过程中，会判断实例是否是垃圾。如果其他对象引用了该实例，垃圾回收器就会认为“该实例正在被使用”，不会将其当作垃圾回收掉。

现在，让我们再回顾一下示例程序。在示例程序中，*pool*字段负责管理已经生成的*BigChar*的实例。因此，只要是*pool*字段管理的*BigChar*的实例，就不会被看作是垃圾，即使该*BigChar*的实例实际上已经不再被*BigString*类的实例所使用。也就是说，只要生成了一个*BigChar*的实例，它就会长期驻留在内存中。在示例程序中，字符串的显示处理很快就结束了，因此不会发生内存不足的问题。但是如果应用程序需要长期运行或是需要以有限的内存来运行，那么在设计程序时，开发人员就必须时刻警惕“不要让被共享的实例被垃圾回收器回收了”。

虽然我们不能显式地删除实例，但我们可以删除对实例的引用。要想让实例可以被垃圾回收器回收掉，只需要显式地将其置于管理对象外即可。例如，只要我们从*HashMap*中移除该实例的*Entry*，就删除了对该实例的引用。

> 内存之外的其他资源

在示例程序中，我们了解到共享实例可以减少内存使用量。一般来说，共享实例可以减少所需资源的使用量。这里的资源指的是计算机中的资源，而内存是资源中的一种。

时间也是一种资源。使用*new*关键字生成实例会花费时间。通过*Flyweight*模式共享实例可以减少使用*new*关键字生成实例的次数。这样，就可以提高程序运行速度。

文件句柄（文件描述符）和窗口句柄等也都是一种资源。在操作系统中，可以同时使用的文件句柄和窗口句柄是有限制的。因此，如果不共享实例，应用程序在运行时很容易就会达到资源极限而导致崩溃。



### 20.5 相关的设计模式

+ ***Proxy*模式**

如果生成实例的处理需要花费较长时间，那么使用*Flyweight*模式可以提高程序的处理速度。

而*Proxy*模式则是通过设置代理提高程序的处理速度。

+ ***Composite*模式**

有时可以使用*Flyweight*模式共享*Composite*模式中的*Leaf*角色。

+ ***Singleton*模式**

在*FlyweightFactory*角色中有时会使用*Singleton*模式。

此外，如果使用了*Singleton*模式，由于只会生成一个*Singleton*角色，因此所有使用该实例的地方都共享同一个实例。在*Singleton*角色的实例中只持有*intrinsic*信息。