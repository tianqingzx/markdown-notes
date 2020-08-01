[[_toc_]]

### 3.1 Template Method模式

**什么是 Template Method 模式**

在父类中定义处理流程的框架，在子类中实现具体处理的模式就称为*Template Method*模式。

### 3.2 示例程序

**类的一览表**

| 名字            | 说明                             |
| :-------------- | -------------------------------- |
| AbstractDisplay | 只实现了display方法的抽象类      |
| CharDisplay     | 实现了open、print、close方法的类 |
| StringDisplay   | 实现了open、print、close方法的类 |
| Main            | 测试程序行为的类                 |

**示例程序类图**

<img src="F:\文档\Typora Files\markdown-notes\images\notes\设计模式\template_method_uml.png" alt="template_method_uml" style="zoom:80%;" />

> AbstractDisplay类

```java
public abstract class AbstractDisplay {
    public abstract void open();
    public abstract void print();
    public abstract void close();
    public final void display() {
        open();
        for (int i=0; i<5; i++) {
            print();
        }
        close();
    }
}
```

> CharDisplay类

| 方法名 | 处理                      |
| ------ | ------------------------- |
| open   | 显示字符串"<<"            |
| print  | 显示构造函数接受的1个字符 |
| close  | 显示字符串">>"            |



```java
public class CharDisplay extends AbstractDisplay {
    private char ch;
    public CharDisplay(char ch) {
        this.ch = ch;
    }
    @Override
    public void open() {
        System.out.print("<<");
    }
    @Override
    public void print() {
        System.out.print(ch);
    }
    @Override
    public void close() {
        System.out.println(">>");
    }
}
```

> StringDisplay类

| 方法名 | 处理                                           |
| ------ | ---------------------------------------------- |
| open   | 显示字符串"+-------+"                          |
| print  | 在构造函数接收的字符串前后分别加上"\|"显示出来 |
| close  | 显示字符串"+-------+"                          |



```java
public class StringDisplay extends AbstractDisplay {
    private String string;
    private int width;
    public StringDisplay(String string) {
        this.string = string;
        this.width = string.getBytes().length;
    }
    @Override
    public void open() {
        printLine();
    }
    @Override
    public void print() {
        System.out.println("|" + string + "|");
    }
    @Override
    public void close() {
        printLine();
    }
    private void printLine() {
        System.out.print("+");
        for (int i=0; i<width; i++) {
            System.out.print("-");
        }
        System.out.println("+");
    }
}
```

> Main类

```java
public class Main {
    public static void main(String[] args) {
        AbstractDisplay d1 = new CharDisplay('H');
        AbstractDisplay d2 = new StringDisplay("Hello World.");
        AbstractDisplay d3 = new StringDisplay("你好，世界。");
        d1.display();
        d2.display();
        d3.display();
    }
}
```

**最终的运行截图：**

<img src="F:\文档\Typora Files\markdown-notes\images\notes\设计模式\template_method.png" alt="template_method" style="zoom:80%;" />

### 3.3Template Method 模式中的登场角色

**AbstractDisplay（抽象类）**

*AbstractClass*角色不仅负责实现模板方法，还负责声明在模板方法中所使用到的抽象方法。这些抽象方法由子类*ConcreteClass*角色负责实现。在示例程序中，由*AbstractDisplay*类扮演此角色。

**ConcreteClass（具体类）**

该角色负责具体实现*AbstractClass*角色中定义的抽象方法。这里实现的方法将会在*AbstractClass*角色的模板方法中被调用，由*CharDisplay*类和*StringDisplay*类扮演此角色。

### 3.4 扩展思路的要点

**可以实现使逻辑处理通用化**

它的有点是由于在父类的模板方法中编写了算法，因此无需在每个子类中再编写算法。

### 3.5 相关的设计模式

+ Factory Method 模式

*Factory Method*模式是将*Template Method*模式用于生成实例的一个典型例子。

+ Strategy 模式

在*Template Method*模式中，可以**使用继承改变程序的行为**。

与此相对的是*Strategy*模式，它可以**使用委托改变程序的行为**。该模式用于替换整个算法。

### 附：

`java.io.InputStream`类使用了*Template Method*模式。可以阅读官方文档（JDK的API参考资料），从中找出需要用`java.io.InputStream`的子类去实现的方法。