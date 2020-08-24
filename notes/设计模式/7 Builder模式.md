# 第7章 Builder模式
### 组装复杂的实例

### 7.1 Builder模式

*Builder*模式主要用于组装具有复杂结构的实例

### 7.2 示例程序

作为示例程序，我们来看一段使用*Builder*模式编写“文档”的程序。这里编写出的文档具有以下结构。

+ **含有一个标题**

+ **含有几个字符串**

+ **含有条目项目**

*Builder*类中定义了决定文档结构的方法，然后*Director*类使用该方法编写一个具体的文档。

*Builder*是抽象类，它并没有进行任何实际的处理，仅仅声明了抽象方法。*Builder*类的子类决定了用来编写文档的具体处理。

在示例程序中，我们定义了以下*Builder*类的子类。

+ ***TextBuilder*类：使用纯文本（普通字符串）编写文档**

+ ***HTMLBuilder*类：使用*HTML*编写文档**

*Director*使用*TextBuilder*类时可以编写纯文本文档；使用*HTMLBuilder*类时可以编写*HTML*文档。

**类的一览表**

| 名字        | 说明                                 |
| ----------- | ------------------------------------ |
| Builder     | 定义了决定文档结构的方法的抽象类     |
| Director    | 编写1个文档的类                      |
| TextBuilder | 使用纯文本（普通字符串）编写文档的类 |
| HTMLBuilder | 使用HTML编写文档的类                 |
| Main        | 测试程序行为的类                     |

**示例程序的类图**

![builder_uml](F:\文档\Typora Files\markdown-notes\images\notes\设计模式\builder_uml.png)

> Builder类

*Builder*类是一个声明了编写文档的方法的抽象类。*makeTitle、makeString、makeTimes*方法分别是编写标题、字符串、条目的方法。*close*方法是完成文档编写的方法。

```java
public abstract class Builder {
    public abstract void makeTitle(String title);
    public abstract void makeString(String str);
    public abstract void makeItems(String[] items);
    public abstract void close();
}
```

> Director类

*Director*类使用*Builder*类中声明的方法来编写文档。

*Director*类的构造函数的参数是*Builder*类型的。但是实际上我们并不会将*Builder*类的实例作为参数传递给*Director*类。这是因为*Builder*类是抽象类，是无法生成其实例的。实际上传递给*Director*类的是*Builder*类的子类（即后面会讲到的*TextBuilder*类和*HTMLBuilder*类等）的实例。而正是这些*Builder*类的子类决定了编写出的文档的形式。

*construct*方法是编写文档的方法。调用这个方法后就会编写文档。*construct*方法中所使用的方法都是在*Builder*类中声明的方法（*construct*的意思是“构建”）。

```java
public class Director {
    private Builder builder;
    public Director(Builder builder) {  // 因为接收的参数是Builder类的子类
        this.builder = builder;  // 所以可以将其保存在builder字段中
    }
    public void construct() {  // 编写文档
        builder.makeTitle("Greeting");  // 标题
        builder.makeString("从早上至下午");  // 字符串
        builder.makeItems(new String[]{  // 条目
                "早上好。",
                "下午好。",
        });
        builder.makeString("晚上");  // 其他字符串
        builder.makeItems(new String[]{  // 其他条目
                "晚上好。",
                "晚安。",
                "再见。",
        });
        builder.close();  // 完成文档
    }
}
```

> TextBuilder类

*TextBuilder*类是*Builder*类的子类，它的功能是使用纯文本编写文档，并以*String*返回结果。

```java
public class TextBuilder extends Builder {
    private StringBuffer buffer = new StringBuffer();  // 文档内容保存在该字段中
    @Override
    public void makeTitle(String title) {  // 纯文本的标题
        buffer.append("==============\n");  // 装饰线
        buffer.append("[" + title + "]\n");  // 为标题添加[]
        buffer.append("\n");  // 换行
    }
    @Override
    public void makeString(String str) {  // 纯文本的字符串
        buffer.append("> " + str + "\n");  // 为字符串添加>
        buffer.append("\n");  // 换行
    }
    @Override
    public void makeItems(String[] items) {  // 纯文本的条目
        for (int i=0; i < items.length; i++) {
            buffer.append("   " + items[i] + "\n");  // 为条目添加
        }
        buffer.append("\n");  // 换行
    }
    @Override
    public void close() {  // 完成文档
        buffer.append("================\n");  // 装饰线
    }
    public String getResult() {  // 完成的文档
        return buffer.toString();  // 将 StringBuffer 变换为 String
    }
}
```

> HTMLBuilder类

*HTMLBuilder*类也是*Builder*类的子类，它的功能是使用*HTML*编写文档，其返回结果是*HTML*文件的名字。

```java
public class HTMLBuilder extends Builder {
    private String filename;  // 文件名
    private PrintWriter writer;  // 用于编写文件的PrintWriter
    @Override
    public void makeTitle(String title) {  // HTML文件的标题
        filename = title + ".html";  // 将标题作为文件名
        try {
            writer = new PrintWriter(new FileWriter(filename));  // 生成PrintWriter
        } catch (IOException e) {
            e.printStackTrace();
        }
        writer.println("<html><head><title>" + title + "</title></head><body>");
        // 输出标题
        writer.println("<h1>" + title + "</h1>");
    }
    @Override
    public void makeString(String str) {  // HTML文件中的字符串
        writer.println("<p>" + str + "</p>");  // 用<p>标签输出
    }
    @Override
    public void makeItems(String[] items) {  // HTML文件中的条目
        writer.println("<ul>");  // 用<ul>和<li>输出
        for (int i=0; i < items.length; i++) {
            writer.println("<li>" + items[i] + "</li>");
        }
        writer.println("</ul>");
    }
    @Override
    public void close() {  // 完成文档
        writer.println("</body></html>");  // 关闭标签
        writer.close();  // 关闭文件
    }
    public String getResult() {  // 编写完成的文档
        return filename;  // 返回文件名
    }
}
```

> Main类

*Main*类是*Builder*模式的测试程序。我们可以使用如下的命令来编写相应格式的文档：

***java Main plain*：编写纯文本文档**

***java Main html*：编写HTML格式的文档**

当我们在命令行中指定参数为***plain***的时候，会将*TextBuilder*类的实例作为参数传递至*Director*类的构造函数中；而若是在命令行中指定参数为***html***的时候，则会将*HTMLBuilder*类的实例作为参数传递至*Director*类的构造函数中。

由于*TextBuilder*和*HTMLBuilder*都是*Builder*的子类，因此*Director*仅仅使用*Builder*的方法即可编写文档。也就是说，***Director*并不关心实际编写文档的到底是*TextBuilder*还是*HTMLBuilder***。

正因为如此，我们必须在*Builder*中声明足够多的方法，以实现编写文档的功能，但并不包括*TextBuilder*和*HTMLBuilder*中特有的方法。

```java
public class Main {
    public static void main(String[] args) {
        if (args.length != 1) {
            usage();
            System.exit(0);
        }
        if (args[0].equals("plain")) {
            TextBuilder textBuilder = new TextBuilder();
            Director director = new Director(textBuilder);
            director.construct();
            String result = textBuilder.getResult();
            System.out.println(result);
        } else if (args[0].equals("html")) {
            HTMLBuilder htmlBuilder = new HTMLBuilder();
            Director director = new Director(htmlBuilder);
            director.construct();
            String filename = htmlBuilder.getResult();
            System.out.println(filename + " 文件编写完成。");
        } else {
            usage();
            System.exit(0);
        }
    }
    public static void usage() {
        System.out.println("Usage: java Main plain    编写纯文本文档");
        System.out.println("Usage: java Main html     编写HTML文档");
    }
}

```

### 7.3 Builder模式中的登场角色

+ ***Builder*（建造者）**

*Builder*角色负责定义用于生成实例的接口（API）。*Builder*角色中准备了用了生成实例的方法。在示例程序中，由*Builder*类扮演此角色。

+ ***ConcreteBuilder*（具体的建造者）**

*ConcreteBuilder*角色是负责实现*Builder*角色接口的类（API）。这里定义了在生成实例时实际被调用的方法。此外，在*ConcreteBuilder*角色中还定义了获取最终生成结果的方法。在示例程序中，由*TextBuilder*类和*HTMLBuilder*类扮演此角色。

+ ***Director（*监工）**

*Director*角色负责使用*Builder*角色的接口（API）来生成实例。它并不依赖于*ConcreteBuilder*角色。为了确保不论*ConcreteBuilder*角色是如何被定义的，*Director*角色都能正常工作，它只调用在*Builder*角色中被定义的方法。在示例程序中，由*Director*类扮演此角色。

+ ***Client*（使用者）**

该角色使用了*Builder*模式。在示例程序中，由*Main*类扮演此角色。

### 7.4 相关的设计模式

+ ***Template Method*模式**

在*Builder*模式中，*Director*角色控制*Builder*角色。

在*Template Method*模式中，父类控制子类。

+ ***Composite*模式**

有些情况下*Builder*模式生成的实例构造成了*Composite*模式。

+ ***Abstract Factory*模式**

*Builder*模式和*Abstract Factory*模式都用于生成复杂的实例。

+ ***Facade*模式**

在*Builder*模式中，*Director*角色通过组合*Builder*角色中的复杂方法向外部提供可以简单生成实例的接口（API）（相当于示例程序中的*construct*方法）。

*Facade*模式中的*Facade*角色则是通过组合内部模块向外部提供可以简单调用的接口（API）。

### 7.5 拓展思路的要点

*Main*类并不知道（没有调用）*Builder*类，它只是调用了*Direct*类的*construct*方法。这样，*Director*类就会开始工作（*Main*类对此一无所知），并完成文档的编写。

另一方面，*Director*类知道*Builder*类，它调用*Builder*类的方法来编写文档，但是它并不知道它“真正”使用的是哪个类。也就是说它并不知道它所使用的类到底是*TextBuilder*类、*HTMLBuilder*类还是其他*Builder*类的子类。不过也没有必要知道，因为*Director*类只使用了*Builder*类的方法，而*Builder*类的子类都已经实现了那些方法。

*Director*类不知道自己使用的究竟是*Builder*类的哪个子类也好。这是因为“只有不知道子类才能替换”。不论是将*TextBuilder*的实例传递给*Director*，还是将*HTMLBuilder*类的实例传递给*Director*，它都可以正常工作，原因正是*Director*类不知道*Builder*类的具体的子类。

正是因为不知道才能够替换，正是因为可以替换，组件才具有高价值。作为设计人员，我们必须时刻关注这种“可替换性”。