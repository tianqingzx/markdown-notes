# 第18章 Memento模式

### 保存对象状态



### 18.1 Memento模式

通过引入表示实例状态的角色，可以在保存和恢复实例时有效地防止对象的封装性遭到破坏。这就是*Memento*模式。

使用*Memento*模式可以实现应用程序的以下功能。

+ ***Undo*（撤销）**

+ ***Redo*（重做）**

+ ***History*（历史记录）**

+ ***Snapshot*（快照）**

*Memento*模式就是一个这样的设计模式，它事先将某个时间点的实例的状态保存下来，之后在有必要时，再将实例恢复至当时的状态。



### 18.2 示例程序

这是一个收集水果和获取金钱数的掷骰子游戏：

+ **游戏是自动进行的**

+ **游戏的主人公通过掷骰子来决定下一个状态**

+ **当骰子点数为1的时候，主人公的金钱会增加**

+ **当骰子点数为2的时候，主人公的金钱会减少**

+ **当骰子点数为6的时候，主人公会得到水果**

+ **主人公没有钱时游戏就会结束**

在程序中，如果金钱增加，为了方便将来恢复状态，我们会生成*Memento*类的实例，将现在的状态保存起来。所保存的数据为当前持有的金钱和水果。如果不断掷出了会导致金钱减少的点数，为了防止金钱变为0而结束游戏，我们会使用*Memento*的实例将游戏恢复至之前的状态。

**类的一览表**

| 包   | 名字    | 说明                                                         |
| ---- | ------- | ------------------------------------------------------------ |
| game | Memento | 表示Gamer状态的类                                            |
| game | Gamer   | 表示游戏主人公的类。它会生成Memento的实例                    |
| 无名 | Main    | 进行游戏的类。它会事先保存Memento的实例，之后会根据需要恢复Gamer的状态 |

**示例程序的类图**

![memento_uml](F:\文档\Typora Files\markdown-notes\images\notes\设计模式\memento_uml.PNG)

> Memento类

*Memento*类是表示*Gamer*（主人公）状态的类。

*Memento*类和*Gamer*类都位于*game*包下。

*Memento*类中有两个字段，即*money*和*fruits*。*money*表示主人公现在所持有的金钱数目，*fruits*表示现在为止所获得的水果。之所以没有将*money*和*fruits*的可见性设为*private*，是因为我们希望同在*game*包下的*Gamer*类可以访问这两个字段。

*GetMoney*方法的作用是获取主人公当前所持有的金钱数目。

*Memento*类的构造函数的可见性并非*public*，因此并不是任何其他类都可以生成*Memento*类的实例。只有在同一个包（本例中是*game*包）下的其他类才能调用*Memento*类的构造函数。具体来说，只有*game*包下的*Gamer*类才能生成*Memento*类的实例。

*addFruit*方法用于添加所获得的水果。该方法的可见性也不是*public*。这是因为只有同一个包下的其他类才能添加水果。因此，**无法从*game*包外部改变*Memento*内部的状态**。

此外，*Memento*类中有*“narrow interface”*和*“wide interface”*这样的注释。

```java
import java.util.ArrayList;
import java.util.List;

public class Memento {
    int money;                      // 所持金钱
    ArrayList fruits;               // 获得水果
    public int getMoney() {         // 获取当前所持金钱(narrow interface)
        return money;
    }
    Memento(int money) {            // 构造函数(wide interface)
        this.money = money;
        this.fruits = new ArrayList();
    }
    void addFruit(String fruit) {   // 添加水果(wide interface)
        fruits.add(fruit);
    }
    List getFruits() {              // 获取当前所持所有水果(wide interface)
        return (List) fruits.clone();
    }
}
```



> Gamer类

*Gamer*类是表示游戏主人公的类。它有3个字段，即所持金钱（*money*）、获得的水果（*fruits*）以及一个随机数生成器（*random*）。而且还有一个名为*fruitsname*的静态字段。

进行游戏的主要方法是*bet*方法。在该方法中，只要主人公没有破产，就会一直掷骰子，并根据骰子结果改变所持有的金钱数目和水果个数。

*createMemento*方法的作用是保存当前的状态（拍摄快照）。在*createMemento*方法中，会根据当前在时间点所持有的金钱和水果生成一个*Memento*类的实例，该实例代表了“当前*Gamer*的状态”，它会被返回给调用者。就如同给对象照了张照片一样，我们将对象现在的状态封存在*Memento*类的实例中。请注意我们只保存了“好吃”的水果。

*restoreMemento*方法的功能与*createMemento*相反，它会根据接收到的*Memento*类的实例来将*Gamer*恢复为以前的状态，仿佛是在游戏中念了一通“复活咒语”一样。

```java
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

public class Gamer {
    private int money;                              // 所持金钱
    private List fruits = new ArrayList();          // 获得的水果
    private Random random = new Random();           // 随机数生成器
    private static String[] fruitsname = {          // 表示水果种类的数组
            "苹果", "葡萄", "香蕉", "橘子",
    };
    public Gamer(int money) {                       // 构造函数
        this.money = money;
    }
    public int getMoney() {                         // 获取当前所持金钱
        return money;
    }
    public void bet() {                             // 投掷骰子进行游戏
        int dice = random.nextInt(6) + 1;    // 掷骰子
        if (dice == 1) {                            // 骰子结果为1时，增加所持金钱
            money += 100;
            System.out.println("所持金钱增加了。");
        } else if (dice == 2) {                     // 骰子结果为2时，所持金钱减半
            money /= 2;
            System.out.println("所持金钱减半了。");
        } else if (dice == 6) {                     // 骰子结果为6时，获得水果
            String f = getFruit();
            System.out.println("获得了水果(" + f + ")。");
            fruits.add(f);
        } else {                                    // 骰子结果为3、4、5则什么都不会发生
            System.out.println("什么都没有发生。");
        }
    }
    public Memento createMemento() {                // 拍摄快照
        Memento m = new Memento(money);
        Iterator it = fruits.iterator();
        while (it.hasNext()) {
            String f = (String) it.next();
            if (f.startsWith("好吃的")) {            // 只保存好的水果
                m.addFruit(f);
            }
        }
        return m;
    }
    public void restoreMemento(Memento memento) {   // 撤销
        this.money = memento.money;
        this.fruits = memento.getFruits();
    }
    public String toString() {                      // 用字符换表示主人公状态
        return "[money = ]" + money + ", fruits = " + fruits + "]";
    }
    private String getFruit() {                     // 获得一个水果
        String prefix = "";
        if (random.nextBoolean()) {
            prefix = "好吃的";
        }
        return prefix + fruitsname[random.nextInt(fruitsname.length)];
    }
}
```



> Main类

*Main*类生成了一个*Gamer*类的实例并进行游戏。它会重复调用*Gamer*的*bet*方法，并显示*Gamer*的所持金钱。

到目前为止，这只是普通的掷骰子游戏，接下来我们来引入*Memento*模式。在变量*memento*中保存了“某个时间点的*Gamer*的状态”。如果运气很好，金钱增加了，会调用*createMemento*方法保存现在的状态；如果运气不好，金钱不足了，就会调用*restoreMemento*方法将钱还给*memento*。

```java
public class Main {
    public static void main(String[] args) {
        Gamer gamer = new Gamer(100);           // 最初的所持金钱数为100
        Memento memento = gamer.createMemento();        // 保存最初的状态
        for (int i = 0; i < 100; i++) {
            System.out.println("====" + i);             // 显示掷骰子的次数
            System.out.println("当前状态：" + gamer);    // 显示主人公现在的状态

            gamer.bet();        // 进行游戏

            System.out.println("所持金钱为" + gamer.getMoney() + "元。");
            // 决定如何处理Memento
            if (gamer.getMoney() > memento.getMoney()) {
                System.out.println("    (所持金钱增加了许多，因此保存游戏当前的状态)");
                memento = gamer.createMemento();
            } else if (gamer.getMoney() < memento.getMoney() / 2) {
                System.out.println("    (所持金钱减少了许多，因此将游戏恢复至以前的状态)");
                gamer.restoreMemento(memento);
            }

            // 等待一段时间
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                System.out.println(e.toString());
            }
            System.out.println("");
        }
    }
}
```

<img src="F:\文档\Typora Files\markdown-notes\images\notes\设计模式\memento.PNG" alt="memento" style="zoom:80%;" />

### 18.3 Memento模式中的登场角色

+ ***Originator*（生成者）**

*Originator*角色会在保存自己的最新状态时生成*Memento*角色。当把以前保存的*Memento*角色传递给*Originator*角色时，它会将自己恢复至生成该*Memento*角色时的状态。在示例程序中，由*Gamer*类扮演此角色。

+ ***Memento*（纪念品）**

*Memento*角色会将*Originator*角色的内部信息整合在一起。在*Memento*角色中虽然保存了*Originator*角色的信息，但它不会向外部公开这些信息。

*Memento*角色有以下两种接口（*API*）

+ ***wide interface*——宽接口（*API*）**

*Memento*角色提供的“宽接口（*API*）”是指所有用于获取恢复对象状态信息的方法的集合。由于宽接口（*API*）会暴露所有*Memento*角色的内部信息，因此能够使用宽接口（*API*）的只有*Originator*角色。

+ ***narrowinterface*——窄接口（*API*）**

*Memento*角色为外部的*Caretaker*角色提供了“窄接口（*API*）”。可以通过窄接口（*API*）获取的*Memento*角色的内部信息非常有限，因此可以有效地防止信息泄露。

通过对外提供以上两种接口（*API*），可以有效地防止对象的封装性被破坏。

在示例程序中，由*Memento*类扮演此角色。

*Originator*角色和*Memento*角色之间有着非常紧密的联系。

+ ***Caretaker*（负责人）**

当*Caretaker*角色想要保存当前的*Originator*角色的状态时，会通知*Originator*角色。*Originator*角色在接收到通知后会生成*Memento*角色的实例并将其返回给*Caretaker*角色。由于以后可能会用*Memento*实例来将*Originator*恢复至原来的状态，因此*Caretaker*角色会一直保存*Memento*实例。在示例程序中，由*Main*类扮演此角色。

不过，*Caretaker*角色只能使用*Memento*角色两种接口（*API*）中的窄接口（*API*），也就是说它无法访问*Memento*角色内部的所有信息。它只是将*Originator*角色生成的*Memento*角色当作一个黑盒子保存起来。

虽然*Originator*角色和*Memento*角色之间是强关联关系，但*Caretaker*角色和*Memento*角色之间是弱关联关系。*Memento*角色对*Caretaker*角色隐藏了自身的内部信息。



### 18.4 拓展思路的要点

> 两种接口（API）和可见性

为了能够实现*Memento*模式中的两套接口（*API*），我们利用了*Java*语言中的可见性。

***Java*语言的可见性**

| 可见性    | 说明                               |
| --------- | ---------------------------------- |
| public    | 所有类都可以访问                   |
| protected | 同一包中的类或是该类的子类可以访问 |
| 无        | 同一包中的类可以访问               |
| private   | 只有该类自身可以访问               |

在*Memento*类的方法和字段中，有带*public*修饰符的，也有不带修饰符的。这表示设计者希望能够进行控制，从而使某些类可以访问这些方法和字段，而其他一些类则无法访问。

**在*Memento*类中使用到的可见性**

| 可见性 | 字段、方法、构造函数 | 哪个类可以访问             |
| ------ | -------------------- | -------------------------- |
| 无     | money                | Memento类、Gamer类         |
| 无     | fruits               | Memento类、Gamer类         |
| public | getMoney             | Memento类、Gamer类、Main类 |
| 无     | Memento              | Memento类、Gamer类         |
| 无     | addFruit             | Memento类、Gamer类         |


在*Memento*类中，只有*getMoney*方法是*public*的，它是一个窄接口（*API*），因此该方法也可以被扮演*Caretaker*角色的*Main*类调用。

由于扮演*Caretaker*角色的*Main*类并不在*game*包下，所以它只能调用*public*的*getMoney*方法。因此，*Main*类无法随意改变*Memento*类的状态。

还有一点需要注意的是，在*Main*类中*Memento*类的构造函数是无法访问的，这就意味着无法像下面这样生成*Memento*类的实例。

`new Memento(100)`

如果像这样编写了代码，在编译代码时编译器就会报错。如果*Main*类中需要用到*Memento*类的实例，可以通过调用*Gamer*类的*createMemento*方法告诉*Gamer*类“我需要保存现在的状态，请生成一个*Memento*类的实例给我”。

如果我们在编程时需要实现“允许有些类访问这个方法，其他类则不能访问这个方法”这种需求，可以像上面这样使用可见性来控制访问权限。



> 需要多少个Memento

在示例程序中，*Main*类只保存了一个*Memento*。如果在*Main*类中使用数组等集合，让它可以保存多个*Memento*类的实例，就可以实现保存各个时间点的对象的状态。



> Memento的有效期限是多久

在示例程序中，我们是在内存中保存*Memento*的，这样并没有什么问题。如果要将*Memento*永远保存在文件中，就会出现有效期限的问题了。

这是因为，假设我们在某个时间点将*Memento*保存在文件中，之后又升级了应用程序版本，那么可能会出现原来保存的*Memento*与当前的应用程序不匹配的情况。



> 划分Caretaker角色和Originator角色的意义

*Caretaker*角色的职责是决定何时拍摄快照，何时撤销以及保存*Memento*角色。

另一方面，*Originator*角色的职责则是生成*Memento*角色和使用接收到的*Memento*角色来恢复自己的状态。

以上就是*Caretaker*角色与*Originator*角色的职责分担。有了这样的职责分担，当我们需要对应以下需求变更时，就可以完全不用修改*Originator*角色。

+ **变更为可以多次撤销**

+ **变更为不仅可以撤销，还可以将现在的状态保存在文件中**



### 18.5 相关的设计模式

+ ***Command*模式**

在使用*Command*模式处理命令时，可以使用*Memento*模式实现撤销功能。

+ ***Protype*模式**

在*Memento*模式中，为了能够实现快照和撤销功能，保存了对象当前的状态。保存的信息只是在恢复状态时所需要的那部分信息。

而在*Protype*模式中，会生成一个与当前实例完全相同的另外一个实例。这两个实例的内容完全一样。

+ ***State*模式**

在*Memento*模式中，是用“实例”表示状态。

而在*State*模式中，则是用“类”表示状态。