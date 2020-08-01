@[toc]

# 第12章 Decorator 模式

### 装饰边框与被装饰物的一致性

### 12.1 Decorator 模式
​	假如现在有一块蛋糕，如果只涂上奶油，其他什么都不加，就是奶油蛋糕。如果加上草莓，就是草莓奶油蛋糕。如果再加上一块黑色巧克力板，上面用白色巧克力写上姓名，然后插上代表年龄的蜡烛，就变成了一块生日蛋糕。
​	不论是蛋糕、奶油蛋糕、草莓蛋糕还是生日蛋糕，他们的核心都是蛋糕。不过，经过涂上奶油，加上草莓等装饰后，蛋糕的味道变得更加甜美了，目的也变得更加明确了。
​	程序中的对象与蛋糕十分相似。首先有一个相当于蛋糕的对象，然后像不断地装饰蛋糕一样地不断地对其增加功能，它就变成了使用目的更加明确的对象。
​	像这样不断地为对象添加装饰的设计模式被称为*Decorator*模式。*Decorator*指的是“装饰物”。

### 12.2 示例程序
​	本章中的示例程序的功能是给文字添加装饰边框。这里所谓的装饰边框是指用“-”“+”“｜”等字符组成的边框。

**类的一览表**

| 名字          | 说明                     |
| ------------- | ------------------------ |
| Display       | 用于显示字符串的抽象类   |
| StringDisplay | 用于显示单行字符串的类   |
| Border        | 用于显示装饰边框的抽象类 |
| SideBorder    | 用于只显示左右边框的类   |
| FullBorder    | 用于显示上下左右边框的类 |
| Main          | 测试程序行为的类         |

**示例程序的类图**

<img src="F:\文档\Typora Files\markdown-notes\images\notes\设计模式\decorator_uml.png" alt="decorator_uml" style="zoom:80%;" />

> Display类

*Display*类是可以显示多行字符串的抽象类。

*getColumns*方法和*getRows*方法分别用于获取横向字符数和纵向行数。它们都是抽象方法，需要字类去实现。*getRowText*方法用于获取指定的某一行的字符串。它也是抽象方法，需要子类去实现。

*show*是显示所有行的字符串的方法。在*show*方法内部，程序会调用*getRows*方法获取行数，调用*getRowText*获取该行需要显示的字符串，然后通过*for*循环语句将所有的字符串显示出来。*show*方法使用了*getRows*和*getRowText*等抽象方法，这属于*Template Method*模式

```java
public abstract class Display {
    public abstract int getColumns();				// 获取横向字符数
    public abstract int getRows();					// 获取纵向行数
    public abstract String getRowText(int row);		// 获取第row行的字符串
    public final void show() {
        for (int i=0; i<getRows(); i++) {
            System.out.println(getRowText(i));
        }
    }
}
```

> StringDisplay类

*StringDisplay*类是用于显示单行字符串的类，由于*StringDisplay*类是*Display*类的子类，因此它肩负着实现*Display*类中声明的抽象方法的重任。

*string*字段中保存的是要显示的字符串。由于*StringDisplay*类只显示一行字符串，因此*getColumns*方法返回*string.getBytes().length*的值，*getRows*方法则返回固定值1。

此外，仅当要获取第0行的内容时*getRowText*方法才会返回*string*字段。以本章开头的蛋糕的比喻来说，*StringDisplay*类就相当于生日蛋糕中的核心蛋糕。

```java
public class StringDisplay extends Display {
    private String string;						// 要显示的字符串
    public StringDisplay(String string) {		// 通过参数传入要显示的字符串
        this.string = string;
    }
    public int getColumns() {					// 得到字符数
        return string.getBytes().length;
    }
    public int getRows() {						// 行数1
        return 1;
    }
    public String getRowText(int row) {			// 仅当row为0时返回值
        if (row == 0) {
            return string;
        } else {
            return null;
        }
    }
}
```

> Border类

*Border*类是装饰边框的抽象类。虽然它所表示的是装饰边框，但它也是*Display*类的子类。

也就是说，通过继承，装饰边框与被装饰物具有了相同的方法。具体而言，*Border*类继承了父类的*getColumns、getRows、getRowText、show*等各方法。从接口（API）角度而言，装饰边框（*Border*）与被装饰物（*Display*）具有相同的方法也就意味着它们具有一致性。

在装饰边框*Border*类中有一个*Display*类型的*display*字段，它表示被装饰物。不过，*display*字段所表示的被装饰物并不仅限于*StringDisplay*的实例。因为，*Border*也是*Display*类的子类，*display*字段所表示的也可能是其他的装饰边框（*Border*类的子类的实例），而且那个边框中也有一个*display*字段。

```java
public abstract class Border extends Display {
    protected Display display;				// 表示被装饰物
    protected Border(Display display) {		// 在生成实例时通过参数指定被装饰物
        this.display = display;
    }
}
```

> SideBorder类

*SideBorder*类是一种具体的装饰边框，是*Border*类的子类。*SideBorder*类用指定的字符（*borderchar*）装饰字符串的左右两侧。可以通过构造函数指定*borderchar*字段。

*SideBorder*类并非抽象类，这是因为它实现了父类中声明的所有抽象方法。

*getColumns*方法是用于获取横向字符数的方法。字符数应当如何计算？只需要在被装饰物的字符数的基础上，再加上两侧边框的字符数即可。那被装饰物的字符数应该如何计算呢？其实只需要调用*display.getColumns()*即可得到被装饰物的字符数。*display*字段的可见性是*protected*，因此*SideBorder*类的子类都可以使用该字段。然后我们再像下面这样，分别加上左右边框的字符数。

`1 + display.getColumns() + 1`

这就是*getColumns*方法的返回值了。当然，写作`display.getColumns() + 2`也是可以的。

在理解了*getColumns*方法的处理方式后，也就可以很快地理解*getRows*方法的处理了。因为*SideBorder*类并不会在字符串的上下两侧添加字符，因此*getRows*方法直接返回*display.getRows()*即可。

那么，*getRowText*方法应该如何实现呢？调用*getRowText*方法可以获取参数指定的那一行的字符数。因此，我们会像下面这样，在*display.getRowText(row)*的字符串两侧，加上*borderchar*这个装饰边框

`borderChar + display.getRowText(row) + borderChar`

这就是*getRowText*方法的返回值（也就是*SideBorder*的装饰效果）。

```java
public class SideBorder extends Border {
    private char borderChar;						// 表示装饰边框的字符
    public SideBorder(Display display, char ch) {	// 通过构造函数指定Display和装饰边框字符
        super(display);
        this.borderChar = ch;
    }
    public int getColumns() {						// 字符数为字符串字符数加上两侧边框字符数
        return 1 + display.getColumns + 1;
    }
    public int getRows() {							// 行数即被装饰物的行数
        return display.getRows();
    }
    public String getRowText(int row) {				// 指定的那一行的字符串为被装饰物的字符串加上两侧的边框的字符
        return borderChar + display.getRowText(row) + borderChar;
    }
}
```

> FullBorder类

*FullBorder*类与*SideBorder*类一样，也是*Border*类的子类。*SideBorder*类会在字符串的左右两侧加上装饰边框，而*FullBorder*类则会在字符串的上下左右都加上装饰边框。不过，在*SideBorder*类中可以指定边框的字符，而在*FullBorder*类中，边框的字符是固定的。

*makeLine*方法可以连续地显示某个指定的字符，它是一个工具方法（为了防止*FullBorder*类外部使用该方法，我们设置它的可见性为*private*）。

```java
public class FullBorder extends Border {
    public FullBorder(Display display) {
        super(display);
    }
    public int getColumns() {		// 字符数为被装饰物的字符数加上两侧边框字符数
        return 1 + display.getColumns() + 1;
    }
    public int getRows() {			// 行数为被装饰物的行数加上上下边框的行数
        return 1 + display.getRows() + 1;
    }
    public String getRowText(int row) {	// 指定的那一行的字符串
        if (row == 0) {				// 下边框
            return "+" + makeLine('-', display.getColumns()) + "+";
        } else if (row == display.getRows() + 1) {	// 上边框
            return "+" + makeLine('-', display.getColumns()) + "+";
        } else {					// 其他边框
            return "|" + display.getRowText(row - 1) + "|";
        }
    }
    private String makeLine(char ch, int count) {	// 生成一个重复count次字符ch的字符串
        StringBuffer buf = new StringBuffer();
        for (int i=0; i<count; i++) {
            buf.append(ch);
        }
        return buf.toString();
    }
}
```

> Main类

*Main*类是用于测试程序行为的类。在*Main*类中一共生成了4个实例，即b1-b4，它们的作用分别如下所示。

b1:将“*Hello, world.*”不加装饰地直接显示出来

b2:在b1的两侧加上装饰边框‘#’

b3:在b2的上下左右加上装饰边框

b4:为“你好，世界。”加上多重边框

```java
public class Main {
    public static void main(String[] args) {
        Display b1 = new StringDisplay("Hello, world.");
        Display b2 = new SideBorder(b1, "#");
        Display b3 = new FullBorder(b2);
        b1.show();
        b2.show();
        b3.show();
        Display b4 = new SideBorder(
        	new FullBorder(
            	new FullBorder(
                	new SideBorder(
                    	new FullBorder(
                        	new StringDisplay("你好，世界。");
                        ), 
                        '*'
                    )
                )
            ), 
            '/'
        );
        b4.show();
    }
}
```

**b3、b2和b1的对象图**



### 12.3 Decorator 模式中的登场角色

在*Decorator*模式中有以下登场角色。

+ ***Component***

  增加功能时的核心角色。以开头的例子来说，装饰前的蛋糕就是*Component*角色。*Component*角色只是定义了蛋糕的接口（API）。在示例程序中，由*Display*类扮演此角色。

+ ***ConcreteComponent***
  该角色是实现了*Component*角色所定义的接口（API）的具体蛋糕。在示例程序中，由*StringDisplay*类扮演此角色。

+ ***Decorator*（装饰物）**
  该角色具有与*Component*角色相同的接口（API）。在它内部保存了被装饰对象——*Component*角色。*Decorator*角色知道自己要装饰的对象。在示例程序中，由*Border*类扮演此角色。

+ ***ConcreteDecorator*（具体的装饰物）**
  该角色是具体的*Decorator*角色。在示例程序中，由*SideBorder*类和*FullBorder*类扮演此角色。

### 12.4 扩展思路的要点

> 接口（API）的透明性

在*Decorator*模式中，装饰边框与被装饰物具有一致性。具体而言，在示例程序中，表示装饰边框的*Border*类是表示被装饰物的*Display*类的子类，这就体现了它们之间的一致性。也就是说，*Border*类（以及它的的子类）与表示被装饰物的*Display*类具有相同的接口（API）。

这样，即使被装饰物被边框装饰起来了，接口（API）也不会被隐藏起来。其他类依然可以调用*getColumns、getRows、getRowText*以及*show*方法。这就是接口（API）的“透明性”。

在示例程序中，实例b4被装饰了多次，但是接口（API）却没有发生任何变化。

得益于接口（API）的透明性，*Decorator*模式中也形成了类似于*Composite*模式中的递归结构。也就是说，装饰边框里面的“被装饰物”实际上又是别的物体的“装饰边框”。就像是剥洋葱时以为洋葱心要出来了，结果却发现还是皮。不过，*Decorator*模式虽然与*Composite*模式一样，都具有递归结构，但是它们的使用目的不同。*Decorator*模式的主要目的是通过添加装饰物来增加对象的功能。

> 在不改变被装饰物的前提下增加功能

在*Decorator*模式中，装饰边框与被装饰物具有相同的接口（API）。虽然接口（API）是相同的，但是越装饰，功能则越多。例如，用*SideBorder*装饰*Display*后，就可以在字符串的左右两侧加上装饰字符。如果再用*FullBorder*装饰，那么就可以在字符串的四周加上边框。此时，我们完全不需要对被装饰的类做任何修改。这样，我们就实现了不修改被装饰的类即可增加功能。

*Decorator*模式使用了委托。对“装饰边框”提出的要求（调用装饰边框的方法）会被转交（委托）给“被装饰物”去处理。以示例程序来说，就是*SideBorder*类的*getColumns*方法调用了*display.getColumns()*。除此以外，*getRows*方法也调用了*display.getRows()*。

> 可以动态地增加功能

*Decorator*模式中用到了委托，它使类之间形成了弱关联关系。因此，不用改变框架代码，就可以生成一个与其他对象具有不同关系的新对象。

> 只需要一些装饰物即可添加许多功能

使用*Decorator*模式可以为程序添加许多功能。只要准备一些装饰边框（*ConcreteDecorator*角色），即使这些装饰边框都只具有非常简单的功能，也可以将它们自由组合成为新的对象。

这就像我们可以自由选择香草味冰淇凌、巧克力冰淇淋、草莓冰淇淋、猕猴桃冰淇淋等各种口味的冰淇淋一样。如果冰淇淋店要为顾客准备所有的冰淇淋成品那真是太麻烦了。因此，冰淇淋店只会准备各种香料，当顾客下单后只需要在冰淇淋上加上各种香料就可以了。不管是香草味，还是咖啡朗姆和开心果的混合口味，亦或是香草味、草莓味和猕猴桃三重口味，顾客想吃什么口味都可以。*Decorator*模式就是可以应对这种多功能对象的需求的一种模式。

> java.io包与Decorator模式

下面我们来谈谈*java.io*包中的类。*java.io*包是用于输入输出（*Input/Output*，简称I/O）的包。这里我们使用了*Decorator*模式。

首先，我们可以像下面这样生成了一个读取文件的实例。

`Reader reader = new FileReader("datafile.txt");`

然后，我们也可以像下面这样在读取文件时将文件内容放入缓冲区。

`Reader reader = new BufferedReader(new FileReader("datafile.txt"));`

这样，在生成*BufferedReader*类的实例时，会指定将文件读取到*FileReader*类的实例中。

再然后，我们也可以像下面这样管理行号。

```java
Reader reader = new LineNumberReader(
	new BufferedReader(
    	new FileReader("datafile.txt");
    )
);
```

无论是*LineNumberReader*类的构造函数还是*BufferedReader*类的构造函数，都可以接收*Reader*类（的子类）的实例作为参数，因此我们可以像上面那样自由地进行各种组合。

我们还可以只管理行号，但不进行缓存处理。

```java
Reader reader = new LineNumberReader(
	new FileReader("datafile.txt");
);
```

接下来，我们还会管理行号，进行缓存，但是我们不从文件中读取数据，而是从网络中读取数据（下面的代码中省略了细节部分和异常处理）。

```java
java.net.Socket socket = new Socket(hostname, portnumber);
...
Reader reader = new LineNumberReader(
	new bufferedReader(
    	new InputStreamReader(
        	socket.getInputStream()
        )
    )
);
```

这里使用的*InputStreamReader*类既接收*getInputStream*方法返回的*InputStream*类的实例作为构造函数的参数，也提供了*Reader*类的接口（API）（*Adapter*模式）。

除了*java.io*包以外，我们还在*javax.swing.border*包为我们提供了可以为界面中的控件添加装饰边框的类。

> 导致增加许多很小的类

*Decorator*模式的一个缺点是会导致程序中增加许多功能类似的很小的类。

### 12.5 相关的设计模式
+ ***Adapter*模式**

*Decorator*模式可以在不改变被装饰物的接口（API）的前提下，为被装饰物添加边框（透明性）

*Adapter*模式用于适配两个不同的接口（API）。

+ ***Stragety*模式**

*Decorator*模式可以像改变被装饰物的边框或是为被装饰物添加多重边框那样，来增加类的功能。

*Stragety*模式通过整体地替换算法来改变类的功能。

### 12.6 延伸阅读：继承和委托中的一致性
“一致性”，即“可以将不同的东西当作同一种东西看待”。

> 继承——父类和子类的一致性	

子类和父类具有一致性。例如：

```java
class Parent {
    ...
    void parentMethod() {
        ...
    }
}
```

此时，*Child*类的实例可以被保存在*Parent*类型的变量中，也可以调用从*Parent*类中继承的方法。

`Parent obj = new Child();`

`obj.parentMethod();`

也就是说，可以像操作*Parent*类的实例一样操作*Child*类的实例。这是将子类当作父类看待的一个例子。

但是，反过来，如果想将父类当作子类一样操作，则需要先进行类型转换。

`Parent obj = new Child();`

`((Child) obj).childMethod();`

> 委托——自己和被委托对象的一致性

使用委托让接口具有透明性时，自己和被委托对象具有一致性。

例如：

```java
class Rose {
    Violet obj = ...
    void method() {
        obj.method();
    }
}

class Violet {
    void method() {
        ...
    }
}
```



*Rose*和*Violet*都有相同的*method*方法。*Rose*将*method*方法的处理委托给了*Violet*。这样，会让人有一种好像这两个类有所关联，又好像没有关联的感觉。

要说有什么奇怪的地方，那就是这两个类虽然都有*method*方法，但是却没有明确地在代码中体现出这个“共通性”。如果要明确地表示*method*方法是共通的，只需要像下面这样编写一个共通的抽象类*Flower*就可以了。

```java
abstract class Flower {
    abstract void method();
}

class Rose extends Flower {
    Violet obj = ...
    void method() {
        obj.method();
    }
}

class Violet extends Flower {
    void method() {
        ...
    }
}
```

或者是像下面这样，让*Flower*作为接口也行。

```java
interface Flower {
    void method();
}

class Rose implements Flower {
    Violet obj = ...
    void method() {
        obj.method();
    }
}

class Violet implements Flower {
    void method() {
        ...
    }
}
```

