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
public class Main {
    public static void main(String[] args) {
        if (args.length != 1) {
            System.out.println("Usage: java Main class.name.of.ConcreteFactory");
            System.out.println("Example 1: java Main listfactory.ListFactory");
            System.out.println("Example 2: java Main tablefactory.TableFactory");
            System.exit(0);
        }
        Factory factory = Factory.getFactory(args[0]);

        Link people = factory.createLink("人民日报", "http://www.people.com.cn/");
        Link gmw = factory.createLink("光明日报", "http://www.gmw.cn/");

        Link us_yahoo = factory.createLink("Yahoo!", "http://www.yahoo.com/");
        Link jp_yahoo = factory.createLink("Yahoo!Japan", "http://www.yahoo.co.jp/");
        Link excite = factory.createLink("Excite", "http://www.excite.com/");
        Link google = factory.createLink("Google", "http://www.google.com/");

        Tray traynews = factory.createTray("日报");
        traynews.add(people);
        traynews.add(gmw);

        Tray trayyahoo = factory.createTray("Yahoo!");
        trayyahoo.add(us_yahoo);
        trayyahoo.add(jp_yahoo);

        Tray traysearch = factory.createTray("检索引擎");
        traysearch.add(trayyahoo);
        traysearch.add(excite);
        traysearch.add(google);

        Page page = factory.createPage("LinkPage", "zx");
        page.add(traynews);
        page.add(traysearch);
        page.output();
    }
}
```

> 具体的工厂：ListFactory类

*ListFactory*类实现了*Factory*类的*createLink*方法、*createTray*方法以及*createPage*方法。当然，各个方法内部只是分别简单地*new*出了*ListLink*类的实例、*ListTray*类的实例以及*ListPage*类的实例（根据实际需求，这里可能需要用*Prototype*模式来进行*clone*）。

```java
public class ListFactory extends Factory {
    @Override
    public Link createLink(String caption, String url) {
        return new ListLink(caption, url);
    }
    @Override
    public Tray createTray(String caption) {
        return new ListTray(caption);
    }
    @Override
    public Page createPage(String title, String author) {
        return new ListPage(title, author);
    }
}
```

> 具体的零件：ListLink类

*ListLink*类是*Link*类的子类。在*ListLink*类中必须实现的方法是在父类中声明的*makeHTML*抽象方法。*ListLink*类使用<li>标签和<a>标签来制作*HTML*片段。这段*HTML*片段也可以与*ListTray*和*ListPage*的结果合并起来。

```java
public class ListLink extends Link {
    public ListLink(String caption, String url) {
        super(caption, url);
    }
    @Override
    public String makeHTML() {
        return "  <li><a href=\"" + url + "\">" + caption + "</a></li>\n";
    }
}
```

> 具体的零件：ListTray类

*ListTray*类是*Tray*类的子类。这里我们重点看一下*makeHTML*方法是如何实现的。*tray*字段中保存了所有需要以*HTML*格式输出的*Item*，而负责将它们以*HTML*格式输出的就是*makeHTML*方法了。

*makeHTML*方法首先使用<li>标签输出标题（*caption*），接着使用<ul>和<li>标签输出每个*Item*。输出的结果先暂时保存在*StringBuffer*中，最后通过*toString*方法将输出结果转换为*String*类型并返回给调用者。

那么，每个*Item*输出为*HTML*格式就是调用每个*Item*的*makeHTML*方法。这里，并不关心变量*item*中保存的实例究竟是*ListLink*的实例还是*ListTray*的实例，只是简单地调用了*item.makeHTML()*语句而已。这里不能使用*switch*语句或if语句去判断变量*item*中保存的实例的类型，否则就是非面向对象编程了。变量*item*是*Item*类型的，而*Item*类又声明了*makeHTML*方法，而且*ListLink*类和*ListTray*类都是*Item*类的子类，因此可以放心地调用。之后*item*会帮我们进行处理。至于*item*究竟进行了什么样的处理，只有*item*的实例（对象）才知道。这就是面向对象的优点。

这里使用的*java.util.Iterator*类与我们在*Iterator*模式一章中所学习的迭代器在功能上是相同的，不过它是*Java*类库中自带的。为了从*java.util.ArrayList*类中得到*java.util.Iterator*，我们调用*iterator*方法。

```java
public class ListTray extends Tray {
    public ListTray(String caption) {
        super(caption);
    }
    @Override
    public String makeHTML() {
        StringBuffer buffer = new StringBuffer();
        buffer.append("<li>\n");
        buffer.append(caption + "\n");
        buffer.append("<ul>\n");
        Iterator it = tray.iterator();
        while (it.hasNext()) {
            Item item = (Item)it.next();
            buffer.append(item.makeHTML());
        }
        buffer.append("</ul>\n");
        buffer.append("</li>\n");
        return buffer.toString();
    }
}
```

> 具体的产品：ListPage类

*ListPage*类是*Page*类的子类。关于*makeHTML*方法，*ListPage*将字段中保存的内容输出为*HTML*格式。作者名（*author*）用*\<address\>*标签输出。

```java
public class ListPage extends Page {
    public ListPage(String title, String author) {
        super(title, author);
    }
    @Override
    public String makeHTML() {
        StringBuffer buffer = new StringBuffer();
        buffer.append("<html><head><title>" + title + "</title></head>\n");
        buffer.append("<body>\n");
        buffer.append("<h1>" + title + "</h1>\n");
        buffer.append("<ul>\n");
        Iterator it = content.iterator();
        while (it.hasNext()) {
            Item item = (Item)it.next();
            buffer.append(item.makeHTML());
        }
        buffer.append("</ul>\n");
        buffer.append("<hr><address>" + author + "</address>");
        buffer.append("</body></html>\n");
        return buffer.toString();
    }
}
```

### 8.3 为示例程序增加其他工厂

之前学习的*listfactory*包的功能是将超链接以条目形式展示出来。现在我们来使用*tablefactory*将链接以表格形式展示出来。

**类的一览表**

| 包           | 名字         | 说明                                                    |
| ------------ | ------------ | ------------------------------------------------------- |
| tablefactory | TableFactory | 表示具体工厂的类（制作TableLink、TableTray、TablePage） |
| tablefactory | TableLink    | 具体零件：表示HTML的超链接的类                          |
| tablefactory | TableTray    | 具体零件：表示含有Link和Tray的类                        |
| tablefactory | TablePage    | 具体产品：表示HTML页面的类                              |

> 具体的工厂：TableFactory类

*TableFactory*类是*Factory*类的子类。*createLink*方法、*createTray*方法以及*createPage*方法的处理是分别生成*TableLink*、*TableTray*、*TablePage*的实例。

```java
public class TableFactory extends Factory {
    public Link createLink(String caption, String url) {
         return new TableLink(caption, url);
    }
    public Tray createTray(String caption) {
        return new TableTray(caption);
    }
    public Page createPage(String title, String author) {
        return new TablePage(title, author);
    }
}
```

> 具体的零件：TableLink类

*TableLink*类是*Link*类的子类。它的*makeHTML*方法的处理是使用<td>标签创建表格的列。在ListLink类中使用的是<li>标签，而这里使用的是<td>标签。

```java
public class TableLink extends Link {
    public TableLink(String caption, String url) {
        super(caption, url);
    }
    public String makeHTML() {
        return "<td><a href=\"" + url "\">" + caption + "</a></td>\n";
    }
}
```

> 具体的零件：TableTray类

*TableTray*类是*Tray*类的子类，其*makeHTML*方法的处理是使用<td>和<table>标签输出*Item*。

```java
public class TableTray extends Tray {
    public TableTray(String caption) {
        super(caption);
    }
    public String makeHTML() {
        StringBuffer buffer = new StringBuffer();
        buffer.append("<td>");
        buffer.append("<table width=\"100%\" border=\"1\"><tr>");
        buffer.append("<td bgcolor=\"#cccccc" align=\"center\" colspan=\"" + tray.size() + "\"><b>" + caption + "</b></td>");
        buffer.append("</tr>\n");
        buffer.append("<tr>\n");
        Iterator it = tray.iterator();
        while (it.hasNext()) {
            Item item = (Item)it.next();
            buffer.append(item.makeHTML());
        }
        buffer.append("</tr></table>");
        buffer.append("</td>");
        return buffer.toString();
    }
}
```

> 具体的产品：TablePage类

*TablePage*类是*Page*类的子类。

```java
public class TablePage extends Page {
    public TablePage(String title, String author) {
        super(title, author);
    }
    public String makeHTML() {
        StringBuffer buffer = new StringBuffer();
        buffer.append("<html><head><title>" + title + "</title></head>\n");
        buffer.append("<body\n>");
        buffer.append("<h1>" + title + "</h1>\n");
        buffer.append("<table width=\"80%\" border=\"3\">\n");
        Iterator it = content.iterator();
        while (it.hasNext()) {
            Item item = (Item)it.next();
            buffer.append("<tr>" + item.makeHTML() + "</tr>");
        }
        buffer.append("</table>\n");
        buffer.append("<hr><address>" + author + "</address>");
        buffer.append("</body></html>\n");
        return buffer.toString();
    }
}
```

### 8.4 Abstract Factory模式中的登场角色

+ ***AbstractProduct*（抽象产品）**

*AbstractProduct*角色负责定义*AbstractFactory*角色所生成的抽象零件和产品的接口（API）。在示例程序中，由*Link*类、*Tray*类和*Page*类扮演此角色。

+ ***AbstractFactory*（抽象工厂）**

*AbstractFactory*角色负责定义用于生成抽象产品的接口（API）。在示例程序中，由*Factory*类扮演此角色。

+ ***Client*（委托者）**

*Client*角色仅会调用*AbstractFactory*角色和*AbstractProduct*角色的接口（API）来进行工作，对于具体的零件、产品和工厂一无所知。在示例程序中，由*Main*类扮演此角色。

+ ***ConcreteProduct*（具体产品）**

*ConcreteProduct*角色负责实现*AbstractProduct*角色的接口（API）。在示例程序中，由以下包中的以下类扮演此角色。

- *listfactory*包：*ListLink*类、*ListTray*类和*ListPage*类

- *tablefactory*包：*TableLink*类、*TableTray*类和*TablePage*类

+ ***ConcreteFactory*（具体工厂）**

*ConcreteFactory*角色负责实现*AbstractFactory*角色的接口（API）。在示例程序中，由以下包中的以下类扮演此角色。

- *listfactory*包：*Listfactory*类

- *tablefactory*包：*Tablefactory*类

### 8.5 拓展思路的要点

> 易于增加具体的工厂

> 难以增加新的零件

例如，我们要在*factory*包中增加一个表示图像的*Picture*零件。在*listfactory*包中，我们必须要做以下修改。

+ 在*ListFactory*中加入*createPicture*方法

+ 新增*ListPicture*类

已经编写完成的具体工厂越多，修改的工作量就会越大。

### 8.6 相关的设计模式

+ ***Builder*模式**

*Abstract Factory*模式通过调用抽象产品的接口（API）来组装抽象产品，生成具有复杂结构的实例。

*Builder*模式则是分阶段地制作复杂实例。

+ ***Factory Method*模式**

有时*Abstract Factory*模式中零件和产品的生成会使用到*Factory Method*模式。

+ ***Composite*模式**

有时*Abstract Factory*模式在制作产品时会使用*Composite*模式。

+ ***Singleton*模式**

有时*Abstract Factory*模式中的具体工厂会使用*Singleton*模式。

### 8.7 延伸阅读：各种生成实例的方法的介绍

在*Java*中可以使用下面这些方法生成实例。

+ **new**

一般我们使用*Java*关键字*new*生成实例。

可以像下面这样生成*Something*类的实例并将其保存在*obj*变量中。

```java
Something obj = new Something();
```

这时，类名（此处的*Something*）会出现在代码中。

+ **clone**

我们也可以使用在*Prototype*模式中学习过的*clone*方法，根据现有的实例复制出一个新的实例。

我们可以像下面在这样根据自身来复制出新的实例（不过不会调用构造函数）。

```java
class Something {
    ...
    public Something createClone() {
        Something obj = null;
        try {
            obj = (Something)clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }
        return obj;
    }
}
```



+ **newInstance**

使用本章中学习过的*java.lang.Class*类的*newInstance*方法可以通过*Class*类的实例生成出*Class*类所表示的实例（会调用无参构造函数）。

在本章的示例程序中，我们已经展示过如何使用*newInstance*了。下面我们再看一个例子。假设我们现在已经有了*Something*类的实例*someobj*，通过下面的表达式可以生成另外一个*Something*类的实例。

```java
someone.getClass().newInstance()
```

实例上，调用*newInstance*方法可能会导致抛出*InstantiationException*异常或是*IllegalAccessException*异常，因此需要将其置于*try...catch*语句块中或是用*throws*关键字指定调用*newInstance*方法的方法可能会抛出的异常。