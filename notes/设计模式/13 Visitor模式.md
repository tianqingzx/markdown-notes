# 第13章 Visitor模式
### 访问数据结构并处理数据

### 13.1 Visitor模式

> Visitor是“访问者”的意思。

在数据结构中保存着许多元素，我们会对这些元素进行“处理”。这时，“处理”代码放在哪里比较好呢？通常的做法是将它们放在表示数据结构的类中。但是，如果“处理”有许多种呢？这种情况下，每当增加一种处理，我们就不得不去修改表示数据结构的类。

在*Visitor*模式中，**数据结构与处理被分离开来**。我们编写一个表示“访问者”的类来访问数据结构中的元素，并把对各元素的处理交给访问者类。这样，当需要增加新的处理时，我们只需要编写新的访问者，然后让数据结构可以接受访问者的访问即可。

### 13.2 示例程序

在示例程序中，我们使用*Composite*模式中用到的那个文件和文件夹的例子作为访问者要访问的数据结构。访问者会访问由文件和文件夹构成的数据结构，然后显示出文件和文件夹的一览。

**类和接口的一览表**

| 名字                    | 说明                                                       |
| ----------------------- | ---------------------------------------------------------- |
| Visitor                 | 表示访问者的抽象类，它访问文件和文件夹                     |
| Element                 | 表示数据结构的接口，它接受访问者的访问                     |
| ListVisitor             | Visitor类的子类，显示文件和文件夹一览                      |
| Entry                   | File类和Directory类的父类，它是抽象类（实现了Element接口） |
| File                    | 表示文件的类                                               |
| Directory               | 表示文件夹的类                                             |
| FileTreatementException | 表示向文件中add时发生的异常的类                            |
| Main                    | 测试程序行为的类                                           |

**示例程序的类图**

![visitor](F:\文档\Typora Files\markdown-notes\images\notes\设计模式\visitor.png)

> Visitor类

*Visitor*类是表示访问者的抽象类。访问者依赖于它所访问的数据结构（即*File*类和*Directory*类）。

*Visitor*类中定义了两个方法，名字都叫*visit*。不过它们接收的参数不同，一个接收*File*类型的参数，另一个接收*Directory*类型的参数。从外部调用*visit*方法时，程序会根据接收的参数的类型自动选择和执行相应的*visit*方法。通常，我们称这种方式为方法的重载。

*visit（File）*是用于访问*File*类的方法，*visit（Directory）*则是用于访问*Directory*类的方法。在*Visitor*模式中，各个类之间的相互调用非常复杂，单看*Visitor*类是无法整体理解该模式的。

```java
public abstract class Visitor {
    public abstract void visit(File file);
    public abstract void visit(Directory directory);
}
```

> Element接口

*Visitor*类是表示访问者的类，而*Element*接口则是接受访问者的访问的接口。

*Element*接口中声明了*accept*方法（*accept*在英文中是“接受”的意思）。该方法的参数是访问者*Visitor*类。

```java
public interface Element {
    public abstract void accept(Visitor v);
}
```

> Entry类

虽然*Entry*类在本质上与*Composite*模式中的*Entry*类是一样的，不过本章中的*Entry*类实现（*implements*）了*Element*接口。这是为了让*Entry*类适用于*Visitor*模式。实际上实现*Element*接口中声明的抽象方法*accept*的是*Entry*类的子类——*File*类和*Directory*类。

*add*方法仅对*Directory*类有效，因此在*Entry*类中，我们让它简单地报错。同样的，用于获取*Iterator*的*iterator*方法也仅对*Directory*类有效，我们也让它简单地报错。

```java
public abstract class Entry implements Element {
    public abstract String getName();
    public abstract int getSize();

    public Entry add(Entry entry) throws FileTreatmentException {
        throw  new FileTreatmentException();
    }
    public Iterator iterator() throws FileTreatmentException {
        throw new FileTreatmentException();
    }
    public String toString() {
        return getName() + "(" + getSize() + ")";
    }
}
```

> File类

*File*类也与*Composite*模式中的File类一样。当然，在*Visitor*模式中要注意理解它是如何实现*accept*接口的。*accept*方法的参数是*Visitor*类，然后*accept*方法的内部处理是*“v.visit(this);”*，即调用了*Visitor*类的*visit*方法。*visit*方法被重载了，此处调用的是*visit(File)*。这是因为这里的*this*是*File*类的实例。

通过调用*visit*方法，可以告诉*Visitor*“正在访问的对象是*File*类的实例*this*”。

```java
public class File extends Entry {
    private String name;
    private int size;
    public File(String name, int size) {
        this.name = name;
        this.size = size;
    }
    @Override
    public String getName() {
        return name;
    }
    @Override
    public int getSize() {
        return size;
    }
    @Override
    public void accept(Visitor v) {
        v.visit(this);
    }
}
```

> Directory类

*Directory*类是表示文件夹的类。与*Composite*模式中的*Directory*类相比，本章中的*Directory*类中增加了下面两个方法。

第一个方法是*iterator*方法。*iterator*方法会返回*Iterator*，我们可以使用它遍历文件夹中的所有目录条目（文件和文件夹）。

第二个方法当然就是*accept*方法了。与*File*类中的*accept*方法调用了*visit(File)*方法一样，*Directory*类中的*accept*方法调用了*visit(Directory)*方法。这样就可以告诉访问者“当前正在访问的是*Directory*类的实例”。

```java
public class Directory extends Entry {
    private String name;
    private ArrayList dir = new ArrayList();
    public Directory(String name) {
        this.name = name;
    }
    @Override
    public String getName() {
        return name;
    }
    @Override
    public int getSize() {
        int size = 0;
        Iterator it = dir.iterator();
        while (it.hasNext()) {
            Entry entry = (Entry)it.next();
            size += entry.getSize();
        }
        return size;
    }
    public Entry add(Entry entry) {
        dir.add(entry);
        return this;
    }
    public Iterator iterator() {
        return dir.iterator();
    }
    @Override
    public void accept(Visitor v) {
        v.visit(this);
    }
}
```

> ListVisitor类

*ListVisitor*类是*Visitor*类的子类，它的作用是访问数据结构并显示元素一览。因为*ListVisitor*类是*Visitor*类的子类，所以它实现了*visit(File)*方法和*visit(Directory)*方法。

*currentdir*字段中保存的是现在正在访问的文件夹名字。*visit(File)*方法在访问者访问文件时会被File类的*accept*方法调用，参数file是所访问的File类的实例。也就是说，*visit(File)*方法是用来实现“对*File*类的实例要进行的处理”的。在本例中，我们实现的处理是先显示当前文件夹的名字（*currentdir*），然后显示间隔符号“/”，最后显示文件名。

*visit(Directory)*方法在访问者访问文件夹时会被*Directory*类的*accept*方法调用，参数*directory*是所访问的*Directory*类的实例。

在*visit(Directory)*方法中实现了“对*Directory*类的实例要进行处理”。

本例中我们是如何实现的呢？与*visit(File)*方法一样，我们先显示当前文件夹的名字，接着调用*iterator*方法获取文件夹的*Iterator*，然后通过*Iterator*遍历文件夹中的所有目录条目并调用它们各自的*accept*方法。由于文件夹中可能存在着许多目录条目，逐一访问会非常困难。

*accept*方法调用*visit*方法，*visit*方法又会调用*accept*方法，这样就形成了非常复杂的递归调用。通常的递归调用是某个方法调用自身，在*Visitor*模式中，则是*accept*方法与*visit*方法之间相互递归调用。

```java
public class ListVisitor extends Visitor {
    private String currentdir = "";
    @Override
    public void visit(File file) {
        System.out.println(currentdir + "/" + file);
    }
    @Override
    public void visit(Directory directory) {
        System.out.println(currentdir + "/" + directory);
        String savedir = currentdir;
        currentdir = currentdir + "/" + directory.getName();
        Iterator it = directory.iterator();
        while (it.hasNext()) {
            Entry entry = (Entry)it.next();
            entry.accept(this);
        }
        currentdir = savedir;
    }
}
```

> FileTreatmentException类

*FileTreatmentException*类与*Composite*模式中的*FileTreatmentException*类完全相同。

```java
public class FileTreatmentException extends RuntimeException {
    public FileTreatmentException() {}
    public FileTreatmentException(String msg) {
        super(msg);
    }
}
```

> Main类

*Main*类与*Composite*模式中的*Main*类基本相同。不同之处仅仅在于，本章中的*Main*类使用了访问者*ListVisitor*类的实例来显示*Directory*中的内容。

在*Composite*模式中，我们调用*printList*方法来显示文件夹中的内容。该方法已经在*Directory*类中被实现了。与之相对，在*Visitor*模式中是在访问者中显示文件夹中的内容。这是因为显示文件夹中的内容也属于对数据结构中的各元素进行的处理。

```java
public class Main {
    public static void main(String[] args) {
        try {
            System.out.println("Making root entries...");
            Directory rootdir = new Directory("root");
            Directory bindir = new Directory("bin");
            Directory tmpdir = new Directory("tmp");
            Directory userdir = new Directory("user");
            rootdir.add(bindir);
            rootdir.add(tmpdir);
            rootdir.add(userdir);
            bindir.add(new File("vi", 10000));
            bindir.add(new File("latex", 20000));
            rootdir.accept(new ListVisitor());

            System.out.println("");
            System.out.println("Making user entries...");
            Directory yuki = new Directory("yuki");
            Directory hanako = new Directory("hanako");
            Directory tomura = new Directory("tomura");
            userdir.add(yuki);
            userdir.add(hanako);
            userdir.add(tomura);
            yuki.add(new File("diary.html", 100));
            yuki.add(new File("Composite.java", 200));
            hanako.add(new File("memo.tex", 300));
            tomura.add(new File("game.doc", 400));
            tomura.add(new File("junk.mail", 500));
            rootdir.accept(new ListVisitor());
        } catch (FileTreatmentException e) {
            e.printStackTrace();
        }
    }
}
```

**运行结果**

<img src="F:\文档\Typora Files\markdown-notes\images\notes\设计模式\visitor_uml.png" alt="visitor_uml" style="zoom:80%;" />

> Visitor与Element之间的相互调用

下面是展示了一个文件夹下有两个文件时，示例程序的处理流程。

1、首先，*Main*类生成*ListVisitor*的实例。在示例程序中，*Main*类还生成了其他的*Directory*类和*File*类的实例，但在本图中省略了。

2、接着，*Main*类调用*Directory*类的*accept*方法。这时传递的参数是*ListVisitor*的实例，但我们在本图中省略了。

3、*Directory*类的实例调用接收到的参数*ListVisitor*的*visit(Directory)*方法。

4、接下来，*ListVisitor*类的实例会访问文件夹，并调用找到的第一个文件的*accept*方法。传递的参数是自身（*this*）。

5、File的实例调用接收到的参数*ListVisitor*的*visit(File)*方法。请注意，这时*ListVisitor*的*visit(Directory)*还在执行中（并非多线程执行，而是表示*visit(Directory)*还存在于调用堆栈（*callstack*）中的意思。在时序图中，表示生命周期的长方形的右侧发生了重叠就说明了这一点）。

6、从*visit(File)*返回到*accept*，接着又从*accept*也返回出来，然后调用另外一个*File*的实例（同一个文件夹下的第二个文件）的*accept*方法。传递的参数是*ListVisitor*的实例*this*。

7、与前面一样，File的实例调用*visit(File)*方法。所有的处理完成后，逐步返回，最后回到*Main*类中的调用*accept*方法的地方。

在阅读时序图时，请注意以下几点。

+ **对于*Directory*类的实例和File类的实例，我们调用了它们的*accept*方法。**

+ **对于每一个*Directory*类的实例和File类的实例，我们只调用了一次它们的*accept*方法**

+ **对于*ListVisitor*的实例，我们调用了它的*visit(Directory)*和*visit(File)*方法**

+ **处理*visit(Directory)*和*visit(File)*的是同一个*ListVisitor*的实例。**

在*Visitor*模式中，*visit*方法将“处理”都集中在*ListVisitor*里面了。

> 13.3 Visitor模式中的登场角色

+ ***Visitor*（访问者）**

*Visitor*角色负责对数据结构中每个具体的元素（*ConcreteElement*角色）声明一个用于访问*xxxxx*的*visit(xxxxx)*方法。*visit(xxxxx)*是用于处理*xxxxx*的方法，负责实现该方法的是*ConcreteVisitor*角色。在示例程序中，由*Visitor*类扮演此角色。

+ ***ConcreteVisitor*（具体的访问者）**

*ConcreteVisitor*角色负责实现*Visitor*角色所定义的接口（API）。它要实现所有的*visit(xxxxx)*方法，即实现如何处理每个*ConcreteElement*角色。在示例程序中，由*ListVisitor*类扮演此角色。如同在*ListVisitor*中，*currentdir*字段的值不断发生变化一样，随着*visit(xxxxx)*处理的进行，*ConcreteVisitor*角色的内部状态也会不断地发生变化。

+ ***Element*（元素）**

*Element*角色表示Visitor角色访问对象。它声明了接受访问者的*accept*方法。*accept*方法接收到的参数是*Visitor*角色。在示例程序中，由*Element*接口扮演此角色。

+ ***ConcreteElement***

*ConcreteElement*角色负责实现*Element*角色所定义的接口（API）。在示例程序中，由*File*类和*Directory*类扮演此角色。

+ ***ObjectStructure*（对象结构）**

*ObjectStructure*角色负责处理*Element*角色的集合。*ConcreteVisitor*角色为每个*Element*角色都准备了处理方法。在示例程序中，由*Directory*类扮演此角色（一人分饰两角）。为了让*ConcreteVisitor*角色可以遍历处理每个*Element*角色，在示例程序中，我们在*Directory*类中实现了*iterator*方法。

### 13.4 拓展思路的要点

> 双重分发

以下是Visitor模式中方法的调用关系：

*accept*(接受)方法的调用方式如下。

`element.accept(visitor);`

而*visit*(访问)方法的调用方式如下。

`visitor.visit(element);`

对比一下这两个方法会发现，它们是相反的关系。*element*接受*visitor*，而*visitor*又访问*element*。

在*Visitor*模式中，*ConcreteElement*和*ConcreteVisitor*这两个角色共同决定了实际进行的处理。这种消息分发的方式一般被称为**双重分发**（*double dispatch*）。

> 为什么要弄得这么复杂

*Visitor*模式的目的是**将处理从数据结构中分离出来**。数据结构很重要，它能将元素集合和处理关联在一起。但是，需要注意的是，保存数据结构与以数据结构为基础进行处理是两种不同的东西。

在示例程序中，我们创建了*ListVisitor*类作为显示文件夹内容的*ConcreteVisitor*角色。通常，*ConcreteVisitor*角色的开发可以独立于*File*类和*Directory*类。也就是说，*Visitor*模式提高了*File*类和*Directory*类**作为组件的独立性**。如果将进行处理的方法定义在*File*类和*Directory*类中，当每次要扩展功能，增加新的“处理”时，就不得不去修改*File*类和*Directory*类。

> 开闭原则——对扩展的开放，对修改关闭

开闭原则（*The Open-Closed Principle，OCP*）。该原则主张类应当是下面这样的。

+ **对扩展是开放的**

+ **对修改是关闭的**

即，**在不修改现有代码的前提下进行扩展**，这就是开闭原则。

> 易于增加ConcreteVisitor角色

> 难以增加ConcreteELement角色

> Visitor工作需要的条件

“在*Visitor*模式中，对数据结构中的元素进行处理的任务被分离出来，交给*Visitor*类负责。这样就实现了数据结构与处理的分离”。但是要达到这个目的是有条件的，那就是*Element*角色必须向*Visitor*角色公开足够多的信息。

例如，在示例程序中，*visit(Directory)*方法需要调用每个目录条目的*accept*方法。为此，*Directory*类必须提供用于获取每个目录条目的*iterator*方法。

访问者只有从数据结构中获取了足够多的信息后才能工作。如果无法获取到这些信息，它就会无法工作。这样做的缺点是，如果公开了不应当公开的信息，将来对数据结构的改良就会变得非常困难。

> 13.5 相关的设计模式

+ ***Iterator*模式**

*Iterator*模式和*Visitor*模式都是在某种数据结构上进行处理。

*Iterator*模式用于逐个遍历保存在数据结构中的元素。

*Visitor*模式用于对保存在数据结构中的元素进行某种特定的处理。

+ ***Composite*模式**

有时访问者所访问的数据结构会使用*Composite*模式。

+ ***Interpreter*模式**

在*Interpreter*模式中，有时会使用*Visitor*模式。例如，在生成了语法树后，可能会使用*Visitor*模式访问语法树的各个节点进行处理。