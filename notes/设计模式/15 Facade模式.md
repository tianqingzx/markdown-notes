# 第15章 Facade模式
### 简单窗口

### 15.1 Facade模式

随着时间的推移，程序中的类会越来越多，而且它们之间相互关联，这会导致程序结构也变得越来越复杂。这个时候可以为这个大型程序准备一个“窗口”，这样我们就不必单独地关注每个类了，只需要简单地对“窗口”提出请求即可。

这个“窗口”就是我们在本章中将要学习的*Facade*模式。使用*Facade*模式可以为互相关联在一起的错综复杂的类整理出高层接口（API）。其中的*Facade*角色可以让系统对外只有一个简单的接口（API）。而且，*Facade*角色还会考虑到系统内部各个类之间的责任关系和依赖关系，按照正确的顺序调用各个类。

### 15.2 示例程序

在示例程序中，我们将要编写简单的*Web*页面。

本来，编写Facade模式的示例程序需要“许多错综复杂地关联在一起的类”。不过在本书中，为了使示例程序更加简短，我们只考虑一个由3个简单的类构成的系统。也就是一个用于从邮件地址中获取用于名字的数据库（Database），一个用于编写HTML文件的类（HtmlWriter），以及一个扮演Facade角色并提供高层接口（API）的类（PageMaker）。

**类的一览表**

| 包        | 名字       | 说明                            |
| --------- | ---------- | ------------------------------- |
| pagemaker | Database   | 从邮件地址中获取用户名的类      |
| pagemaker | HtmlWriter | 编写HTML文件的类                |
| pagemaker | PageMaker  | 根据邮件地址编写该用户的Web页面 |
| pagemaker | Main       | 测试程序行为的类                |

**示例程序类图**

<img src="F:\文档\Typora Files\markdown-notes\images\notes\设计模式\facade_uml.png" alt="facade_uml" style="zoom:80%;" />

> Database类

*Database*类可获取指定数据库名（如*maildata*）所对应的*Properties*的实例。我们无法生成该类的任何实例，只能通过它的*getProperties*静态方法获取*Properties*的实例。

```java
public class Database {
    private Database() {  // 防止外部new出Database的实例，所以声明为private
    }
    public static Properties getProperties(String dbname) {  // 根据数据库名获取Properties
        String filename = dbname + ".txt";
        Properties prop = new Properties();
        try {
            prop.load(new FileInputStream(filename));
        } catch (IOException e) {
            System.out.println("Warning: " + filename + " is not found.");
            e.printStackTrace();
        }
        return prop;
    }
}
```

> HtmlWriter类

*HtmlWriter*类用于编写简单的*Web*页面。我们在生成*HtmlWriter*类的实例时赋予其*Writer*，然后使用该*Writer*输出*HTML*。

*title*方法用于输出标题；*paragraph*方法用于输出段落；*link*方法用于输出超链接；*mailto*方法用于输出邮件地址链接；*close*方法用于结束*HTML*的输出。

该类中隐藏着一个限制条件，那就是必须首先调用*title*方法。窗口类*PageMaker*使用*HtmlWriter*类时必须严格遵守这个限制条件。

```java
public class HtmlWriter {
    private Writer writer;
    public HtmlWriter(Writer writer) {
        this.writer = writer;
    }
    public void title(String title) throws IOException {  // 输出标题
        writer.write("<html>");
        writer.write("<head>");
        writer.write("<title>" + title + "</title>");
        writer.write("</head>");
        writer.write("<body>\n");
        writer.write("<h1>" + title + "</h1>\n");
    }
    public void paragraph(String msg) throws IOException {  // 输出段落
        writer.write("<p>" + msg + "</p>\n");
    }
    public void link(String href, String caption) throws IOException {  // 输出超链接
        paragraph("<a href=\"" + href + "\">" + caption + "</a>");
    }
    public void mailTo(String mailAddr, String username) throws IOException {  // 输出邮件地址
        link("mailTo:" + mailAddr, username);
    }
    public void close() throws IOException {  // 结束输出HTML
        writer.write("</body>");
        writer.write("</html>\n");
        writer.close();
    }
}
```

> PageMaker类

*PageMaker*类使用*Database*类和*HtmlWriter*类来生成指定用户的Web页面。

在该类中定义的方法只有一个，那就是*public*的*makeWelcomePage*方法。该方法会根据指定的邮件地址和文件名生成相应的*Web*页面。

*PageMaker*类一手包办了调用*HtmlWriter*类的方法这一工作。对外部，它只提供了*makeWelcomePage*接口。这就是一个简单窗口。

```java
public class PageMaker {
    private PageMaker() {  // 防止外部new出PageMaker的实例，所以声明为private方法
    }
    public static void makeWelcomePage(String mailAddr, String filename) {
        try {
            Properties mailProp = Database.getProperties("mailData");
            String username = mailProp.getProperty(mailAddr);
            HtmlWriter writer = new HtmlWriter(new FileWriter(filename));
            writer.title("Welcome to " + username + "'s page!");
            writer.paragraph(username + "欢迎来到" + username + "的主页。");
            writer.paragraph("等着你的邮件哦！");
            writer.mailTo(mailAddr, username);
            writer.close();
            System.out.println(filename + " is created for " + mailAddr + " (" + username + ")");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

> Main类

*Main*类使用了*Pagemaker*包中的*PageMaker*类，具体内容只有下面这一行。

`PageMaker.makeWelcomePage(“zxxxxxx@qq.com”, “welcome.html”);`

它会获取zxxxx@qq.com的名字，然后编写出一个名为*welcome.html*的*Web*页面。

```java
public class Main {
    public static void main(String[] args) {
        PageMaker.makeWelcomePage("hyuki@hyuki.com", "welcome.html");
    }
}
```

### 15.3 Facade模式中的登场角色

+ ***Facade*（窗口）**

*Facade*角色是代表构成系统的许多其他角色的“简单窗口”。*Facade*角色向系统外部提供高层接口（API）。在示例程序中，由*PageMaker*类扮演此角色。

+ **构成系统的许多其他角色**

这些角色各自完成自己的工作，它们并不知道*Facade*角色。*Facade*角色调用其他角色进行工作，但是其他角色不会调用*Facade*角色。在示例程序中，由*Database*类和*HtmlWriter*类扮演此角色。

+ ***Client*（请求者）**

*Client*角色负责调用*Facade*角色中。在示例程序中，由*Main*类扮演此角色。

**Facade模式的类图**

<img src="F:\文档\Typora Files\markdown-notes\images\notes\设计模式\facade_uml_2.png" alt="facade_uml_2" style="zoom:80%;" />

### 15.4 拓展思路的要点

> Facade角色到底做什么工作

*Facade*模式可以让复杂的东西看起来简单。这里说到的“复杂的东西”到底是什么呢？其实就是在后台工作的这些类之间的关系和它们的使用方法。使用*Facade*模式可以让我们不必在意这些复杂的东西。

这里的重点是接口（API）变少了，这意味着程序与外部的关联关系弱化了，这样更容易使我们的包（类的集合）作为组件被复用。

> 递归地使用Facade模式

假设现在有几个持有*Facade*角色的类的集合。那么，我们可以通过整合这几个集合来引入新的*Facade*角色。也就是说，我们可以递归地使用*Facade*模式。

在超大系统中，往往都含有非常多的类和包。如果我们在每个关键的地方都使用*Facade*模式，那么系统的维护就会变得轻松很多。

### 15.5 相关的设计模式

+ ***Abstract Factory*模式**

可以将*Abstract Factory*模式看作生成复杂实例时的*Facade*模式。因为它提供了“要想生成这个实例只需要调用这个方法就OK了”的简单接口。

+ ***Singleton*模式**

有时会使用*Singleton*模式创建*Facade*角色。

+ ***Mediator*模式**

在*Facade*模式中，*Facade*角色单方面地使用其他角色来提供高层接口（API）。

而在*Mediator*模式中，*Mediator*角色作为*Colleague*角色间的仲裁者负责调停。可以说，***Facade*模式是单向的，而*Mediator*角色是双向的**。