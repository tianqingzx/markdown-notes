@[toc]

# 第8章 Abstract Factory模式

### 将关联零件组装成产品

### 8.1 Abstract Factory模式

*Abstract*的意思是“抽象的”，Factory的意思是“工厂”。将它们组合起来我们就可以知道*Abstract Factory*表示“抽象工厂”的意思。

通常，我们不会将“抽象的”这个词与“工厂”这个词联系到一起。所谓工厂，是将零件组装成产品的地方，这是一项具体的工作。

在*Abstract Factory*模式中，不仅有“抽象工厂”，还有“抽象零件”和“抽象产品”。**抽象工厂的工作是将“抽象零件”组装为“抽象产品”**。

“抽象”这个词的具体含义，指的是“不考虑具体怎么实现，而是仅关注接口（API）”的状态。例如，抽象方法（*Abstract Method*）并不定义方法的具体实现，而是仅仅只确定了方法的名字和签名（参数的类型和个数）。

在*Abstract Factory*模式中将会出现抽象工厂，它会将抽象零件组装为抽象产品。也就是说，**我们并不关心零件的具体实现，而是只关心接口（API）。我们仅使用该接口（API）将零件组装成为产品**。

在*Template Method*模式和*Builder*模式中，子类这一层负责方法的具体实现。在*Abstract Factory*模式中也是一样的。在子类这一层中有具体的工厂，它负责将具体的零件组装成为具体的产品。

### 8.2 示例程序

在示例程序中，类被划分为以下3个包。
+ **factory包：包含抽象工厂、零件、产品的包**

+ **无名包：包含Main类的包**

+ **listfactory包：包含具体工厂、零件、产品的包**

下面是类的一览表和UML类图，上面是抽象工厂，下面是具体工厂。

**类的一览表**

| 包          | 名字        | 说明                                                 |
| ----------- | ----------- | ---------------------------------------------------- |
| factory     | Factory     | 表示抽象工厂的类（制作Link、Tray、Page）             |
| factory     | Item        | 方便统一处理Link和Tray的类                           |
| factory     | Link        | 抽象零件：表示HTML的链接的类                         |
| factory     | Tray        | 抽象零件：表示含有Link和Tray的类                     |
| factory     | Page        | 抽象零件：表示HTML页面的类                           |
| 无名        | Main        | 测试程序行为的类                                     |
| listfactory | ListFactory | 表示具体工厂的类（制作ListLink、ListTray、ListPage） |
| listfactory | ListLink    | 具体零件：表示HTML的链接的类                         |
| listfactory | ListTray    | 具体零件：表示含有Link和Tray的类                     |
| listfactory | ListPage    | 具体零件：表示HTML页面的类                           |

**类图**

![abstract_factory_uml](F:\文档\Typora Files\markdown-notes\images\notes\设计模式\abstract_factory_uml.png)

编译方法如下：

```
java Main.java listfactory/ListFactory.java
```

在之前的示例程序中，只要我们编译了*Main.java*，其他所有必要的类都会被编译。但是，这次我们编译*Main.java*时，只有*Factory.java、Item.java、Link.java、Tray.java、Page.java*会被编译，*ListFactory.java、ListLink.java、ListTray.java、ListPage.java*则不会被编译。这是因为*Main*类只使用了*factory*包，没有直接使用*listfactory*包。因此，我们需要在编译时加上参数来编译*listfactory/ListFactory.java*（这样，*ListFactory.java、ListLink.java、ListTray.java、ListPage.java*就都会被编译）。

> 抽象的零件：Item类

*Item*类是*Link*类和*Tray*类的父类（*Item*有“项目”的意思）。这样，*Link*类和*Tray*类就具有可替换性了。

*caption*字段表示项目的“标题”。

*makeHTML*方法是抽象方法，需要子类来实现这个方法。该方法会返回*HTML*文件的内容（需要子类去实现）。

```java
public abstract class Item {
    protected String caption;
    public Item(String caption) {
        this.caption = caption;
    }
    public abstract String makeHTML();
}
```

> 抽象的零件：Link类

*Link*类是抽象地表示*HTML*的超链接的类。

*url*字段中保存的是超链接所指向的地址。乍一看，在*Link*类中好像一个抽象方法都没有，但实际上并非如此。由于*Link*类中没有实现父类（*Item*类）的抽象方法（*makeHTML*），因此它也是抽象类。

```java
public abstract class Link extends Item {
    protected String url;
    public Link(String caption, String url) {
        super(caption);
        this.url = url;
    }
}
```

> 抽象的零件：Tray类

*Tray*类表示的是一个含有多个*Link*类和*Tray*类的容器（*Tray*有托盘的意思。）

*Tray*类使用*add*方法将*Link*类和*Tray*类集合在一起。为了表示集合的对象是“*Link*类和*Tray*类”，我们设置*add*方法的参数为*Link*类和*Tray*类的父类*Item*类。

虽然*Tray*类也继承了*Item*类的抽象方法*makeHTML*，但它并没有实现该方法。因此，*Tray*类也是抽象类。

```java
public abstract class Tray extends Item {
    protected ArrayList<Item> tray = new ArrayList<>();
    public Tray(String caption) {
        super(caption);
    }
    public void add(Item item) {
        tray.add(item);
    }
}
```

> 抽象的产品：Page类

*Page*类是抽象地表示HTML页面的类。如果将*Link*和*Tray*比喻成抽象的“零件”，那么*Page*类就是抽象的“产品”。*title*和*author*分别是表示页面标题和页面作者的字段。作者的名字通过参数传递给*Page*类的构造函数。

可以使用*add*方法向页面中增加*Item*（即*Link*或*Tray*）。增加的*Item*将会在页面中显示出来。

*output*方法首先根据页面标题确定文件名，接着调用*makeHTML*方法将自身保存的*HTML*内容写入到文件中。

其中，我们可以去掉如下语句（1）中的*this*，将其写为如下语句（2）那样。

```java
writer.write(this.makeHTML());			...(1)
writer.write(makeHTML());				...(2)
```

为了强调调用的是*Page*类自己的*makeHTML*方法，我们显式地加上了*this*。这里调用的*makeHTML*方法是一个抽象方法。*output*方法是一个简单的*Template Method*模式的方法。

```java
public abstract class Page {
    protected String title;
    protected String author;
    protected ArrayList<Item> content = new ArrayList<>();
    public Page(String title, String author) {
        this.title = title;
        this.author = author;
    }
    public void add(Item item) {
        content.add(item);
    }
    public void output() {
        try {
            String filename = title + ".html";
            Writer writer = new FileWriter(filename);
            writer.write(this.makeHTML());
            writer.close();
            System.out.println(filename + "编写完成。");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public abstract String makeHTML();
}
```

> 抽象的工厂：Factory类

代码中的*getFactory*方法可以根据指定的类名生成具体工厂的实例。例如，可以像下面这样，将参数*classname*指定为具体工厂的类名所对应的字符串。

`“listfactory.ListFactory”`

*getFactory*方法通过**调用*Class*类的*forName*方法来动态地读取类信息，接着使用*newInstance*方法生成该类的实例**，并将其作为返回值返回给调用者。

*Class*类属于*java.lang*包，是用来表示类的类。*Class*类包含于*Java*的标准类库中。*forName*是*java.lang.Class*的类方法（静态方法），*newInstance*则是*java.lang.Class*的实例方法。

请注意，虽然*getFactory*方法生成的是具体工厂的实例，但是返回值的类型是抽象工厂类型。

*createLink、createTray、createPage*等方法是用于在抽象工厂中生成零件和产品的方法。这些方法都是抽象方法，具体的实现被交给了*Factory*类的子类。不过，这里确定了方法的名字和签名。

```java
public abstract class Factory {
    public static Factory getFactory(String classname) {
        Factory factory = null;
        try {
            factory = (Factory)Class.forName(classname).newInstance();
        } catch (IllegalAccessException | InstantiationException | ClassNotFoundException e) {
            System.err.println("没有找到 " + classname + " 类。");
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return factory;
    }
    public abstract Link createLink(String caption, String url);
    public abstract Tray createTray(String caption);
    public abstract Page createPage(String title, String author);
}
```

> 使用工厂将零件组装称为产品：Main类

*Main*类使用抽象工厂生产零件并将零件组装成产品。*Main*类中只引入了*factory*包，从这一点可以看出，该类并没有使用任何具体零件、产品和工厂。

具体工厂的类名是通过命令行来指定的。例如，如果要使用*listfactory*包中的*ListFactory*类，可以在命令行中输入以下命令。

`java Main listfactory.ListFactory`

*Main*类会使用*getFactory*方法生成该参数（*arg[0]*）对应的工厂，并将其保存在*factory*变量中。

之后，*Main*类会使用*factory*生成*Link*和*Tray*，然后将*Link*和*Tray*都放入*Tray*中，最后生成*Page*并将生成结果输出至文件。

```java

```

> 具体的工厂：ListFactory类

ListFactory类实现了Factory类的createLink方法、createTray方法以及createPage方法。当然，各个方法内部只是分别简单地new出了ListLink类的实例、ListTray类的实例以及ListPage类的实例（根据实际需求，这里可能需要用Prototype模式来进行clone）。

```java

```

> 具体的零件：ListLink类

ListLink类是Link类的子类。在ListLink类中必须实现的方法是在父类中声明的makeHTML抽象方法。ListLink类使用<li>标签和<a>标签来制作HTML片段。这段HTML片段也可以与ListTray和ListPage的结果合并起来。

```java

```

> 具体的零件：ListTray类

ListTray类是Tray类的子类。这里我们重点看一下makeHTML方法是如何实现的。tray字段中保存了所有需要以HTML格式输出的Item，而负责将它们以HTML格式输出的就是makeHTML方法了。

makeHTML方法首先使用<li>标签输出标题（caption），接着使用<ul>和<li>标签输出每个Item。输出的结果先暂时保存在StringBuffer中，最后通过toString方法将输出结果转换为String类型并返回给调用者。

那么，每个Item输出为HTML格式就是调用每个Item的makeHTML方法。这里，并不关心变量item中保存的实例究竟是ListLink的实例还是ListTray的实例，只是简单地调用了item.makeHTML()语句而已。这里不能使用switch语句或if语句去判断变量item中保存的实例的类型，否则就是非面向对象编程了。变量item是Item类型的，而Item类又声明了makeHTML方法，而且ListLink类和ListTray类都是Item类的子类，因此可以放心地调用。之后item会帮我们进行处理。至于item究竟进行了什么样的处理，只有item的实例（对象）才知道。这就是面向对象的优点。

这里使用的java.util.Iterator类与我们在Iterator模式一章中所学习的迭代器在功能上是相同的，不过它是Java类库中自带的。为了从java.util.ArrayList类中得到java.util.Iterator，我们调用iterator方法。

```java

```

> 具体的产品：ListPage类

ListPage类是Page类的子类。关于makeHTML方法，ListPage将字段中保存的内容输出为HTML格式。作者名（author）用\<address\>标签输出。

```java

```

### 8.3 为示例程序增加其他工厂

之前学习的listfactory包的功能是将超链接以条目形式展示出来。现在我们来使用tablefactory将链接以表格形式展示出来。

**类的一览表**

| 包           | 名字         | 说明                                                    |
| ------------ | ------------ | ------------------------------------------------------- |
| tablefactory | TableFactory | 表示具体工厂的类（制作TableLink、TableTray、TablePage） |
| tablefactory | TableLink    | 具体零件：表示HTML的超链接的类                          |
| tablefactory | TableTray    | 具体零件：表示含有Link和Tray的类                        |
| tablefactory | TablePage    | 具体产品：表示HTML页面的类                              |

> 具体的工厂：TableFactory类

TableFactory类是Factory类的子类。createLink方法、createTray方法以及createPage方法的处理是分别生成TableLink、TableTray、TablePage的实例。

> 具体的零件：TableLink类

TableLink类是Link类的子类。它的makeHTML方法的处理是使用<td>标签创建表格的列。在ListLink类中使用的是<li>标签，而这里使用的是<td>标签。

> 具体的零件：TableTray类

TableTray类是Tray类的子类，其makeHTML方法的处理是使用<td>和<table>标签输出Item。

> 具体的产品：TablePage类

TablePage类是Page类的子类。

### 8.4 Abstract Factory模式中的登场角色

+ AbstractProduct（抽象产品）

AbstractProduct角色负责定义AbstractFactory角色所生成的抽象零件和产品的接口（API）。在示例程序中，由Link类、Tray类和Page类扮演此角色。

+ AbstractFactory（抽象工厂）

AbstractFactory角色负责定义用于生成抽象产品的接口（API）。在示例程序中，由Factory类扮演此角色。

+ Client（委托者）

Client角色仅会调用AbstractFactory角色和AbstractProduct角色的接口（API）来进行工作，对于具体的零件、产品和工厂一无所知。在示例程序中，由Main类扮演此角色。

+ ConcreteProduct（具体产品）

ConcreteProduct角色负责实现AbstractProduct角色的接口（API）。在示例程序中，由以下包中的以下类扮演此角色。

- listfactory包：ListLink类、ListTray类和ListPage类

- tablefactory包：TableLink类、TableTray类和TablePage类

+ ConcreteFactory（具体工厂）

ConcreteFactory角色负责实现AbstractFactory角色的接口（API）。在示例程序中，由以下包中的以下类扮演此角色。

- listfactory包：Listfactory类

- tablefactory包：Tablefactory类

### 8.5 拓展思路的要点

> 易于增加具体的工厂

> 难以增加新的零件

例如，我们要在factory包中增加一个表示图像的Picture零件。在listfactory包中，我们必须要做以下修改。

+ 在ListFactory中加入createPicture方法

+ 新增ListPicture类

已经编写完成的具体工厂越多，修改的工作量就会越大。

### 8.6 相关的设计模式

+ Builder模式

Abstract Factory模式通过调用抽象产品的接口（API）来组装抽象产品，生成具有复杂结构的实例。

Builder模式则是分阶段地制作复杂实例。

+ Factory Method模式

有时Abstract Factory模式中零件和产品的生成会使用到Factory Method模式。

+ Composite模式

有时Abstract Factory模式在制作产品时会使用Composite模式。

+ Singleton模式

有时Abstract Factory模式中的具体工厂会使用Singleton模式。

### 8.7 延伸阅读：各种生成实例的方法的介绍

在Java中可以使用下面这些方法生成实例。

+ new

一般我们使用Java关键字new生成实例。

可以像下面这样生成Something类的实例并将其保存在obj变量中。

```java
Something obj = new Something();
```

这时，类名（此处的Something）会出现在代码中。

+ clone

我们也可以使用在Prototype模式中学习过的clone方法，根据现有的实例复制出一个新的实例。

我们可以像下面在这样根据自身来复制出新的实例（不过不会调用构造函数）。
	。。。

+ newInstance

使用本章中学习过的java.lang.Class类的newInstance方法可以通过Class类的实例生成出Class类所表示的实例（会调用无参构造函数）。

在本章的示例程序中，我们已经展示过如何使用newInstance了。下面我们再看一个例子。假设我们现在已经有了Something类的实例someobj，通过下面的表达式可以生成另外一个Something类的实例。

```java
someone.getClass().newInstance()
```

实例上，调用newInstance方法可能会导致抛出InstantiationException异常或是IllegalAccessException异常，因此需要将其置于try...catch语句块中或是用throws关键字指定调用newInstance方法的方法可能会抛出的异常。