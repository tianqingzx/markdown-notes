# 第14章 Chain of Responsibility模式
### 推卸责任

### 14.1 Chain of Responsibility模式

当外部请求程序进行某个处理，但程序暂时无法直接决定由哪个对象负责处理时，就需要推卸责任。这种情况下，我们可以考虑将多个对象组成一条责任连，然后按照它们在职责链上的顺序一个一个地找出到底应该谁来负责处理。

这种模式被称为*Chain of Responsibility*模式。*Responsibility*有“责任”的意思，在汉语中，该模式称为“责任链”。

使用*Chain of Responsibility*模式可以弱化“请求方”和“处理方”之间的关联关系，让双方各自成为独立可复用的组件。此外，程序还可以应对其他需求，如根据情况不同，负责处理的对象也会发生变化的这种需求。

### 14.2 示例程序

**类的一览表**

| 名字           | 说明                                                 |
| -------------- | ---------------------------------------------------- |
| Trouble        | 表示发生的问题的类。它带有问题编号（number）         |
| Support        | 用来解决问题的抽象类                                 |
| NoSupport      | 用来解决问题的具体类（永远“不处理问题”）             |
| LimitSupport   | 用来解决问题的具体类（仅解决编号小于指定编号的问题） |
| OddSupport     | 用来解决问题的具体类（仅解决奇数编号的问题）         |
| SpecialSupport | 用来解决问题的具体类（仅解决指定编号的问题）         |
| Main           | 制作Support的职责链，制造问题并测试程序行为          |

**类图**

![chain_of_responsibility_uml](F:\文档\Typora Files\markdown-notes\images\notes\设计模式\chain_of_responsibility_uml.png)

> Trouble类

*Trouble*类是表示发生的问题的类。*number*是问题的编号。通过*getNumber*方法可以获取问题编号。

```java
public class Trouble {
    private int number;
    public Trouble(int number) {
        this.number = number;
    }
    public int getNumber() {
        return number;
    }
    public String toString() {
        return "[Trouble " + number + "]";
    }
}
```

> Support类

*Support*类是用来解决问题的抽象类，它是责任链上的对象。

*next*字段中指定了要推卸给的对象。可以通过*setNext*方法设定该对象。

*resolve*方法是需要子类去实现的抽象方法。如果*resolve*返回*true*，则表示问题已经被处理，如果返回*false*则表示问题还没有被处理（即需要被推卸给下一个对象）。*Resolve*有“解决”的意思。

*support*方法会调用*resolve*方法，如果*resolve*方法会返回*false*，则*support*方法会将问题转交给下一个对象。如果已经到达责任链中的最后一个对象，则表示没有人处理问题，将会显示出处理失败的相关信息。

另外，*support*方法调用了抽象方法*resolve*，因此它属于*Template Method*模式。

```java
public abstract class Support {
    private String name;
    private Support next;
    public Support(String name) {
        this.name = name;
    }
    public Support setNext(Support next) {
        this.next = next;
        return next;
    }
    public final void support(Trouble trouble) {
        if (resolve(trouble)) {
            done(trouble);
        } else if (next != null) {
            next.support(trouble);
        } else {
            fail(trouble);
        }
    }
    public String toString() {
        return "[" + name + "]";
    }
    protected abstract boolean resolve(Trouble trouble);
    protected void done(Trouble trouble) {
        System.out.println(trouble + " is resolved by " + this + ".");
    }
    protected void fail(Trouble trouble) {
        System.out.println(trouble + " cannot be resolved.");
    }
}
```

> NoSupport类

*NoSupport*类是*Support*类的子类。*NoSupport*类的*resolve*方法总是返回*false*。即它是一个永远“不解决问题”的类。

```java
public class NoSupport extends Support {
    public NoSupport(String name) {
        super(name);
    }
    @Override
    protected boolean resolve(Trouble trouble) {
        return false;
    }
}
```

> LimitSupport类

*LimitSupport*类解决编号小于*limit*值的问题。*resolve*方法在判断编号小于*limit*值后，只是简单地返回*true*，但实际上这里应该是解决问题的代码。

```java
public class LimitSupport extends Support {
    private int limit;
    public LimitSupport(String name, int limit) {
        super(name);
        this.limit = limit;
    }
    protected boolean resolve(Trouble trouble) {
        if (trouble.getNumber() < limit) {
            return true;
        } else {
            return false;
        }
    }
}
```

> OddSupport类

*OddSupport*类解决奇数编号的问题。

```java
public class OddSupport extends Support {
    public OddSupport(String name) {
        super(name);
    }
    @Override
    protected boolean resolve(Trouble trouble) {
        if (trouble.getNumber() % 2 == 1) {
            return true;
        } else {
            return false;
        }
    }
}
```

> SpecialSupport类

*SpecialSupport*类只解决指定编号的问题。

```java
public class SpecialSupport extends Support {
    private int number;
    public SpecialSupport(String name, int number) {
        super(name);
        this.number = number;
    }
    @Override
    protected boolean resolve(Trouble trouble) {
        if (trouble.getNumber() == number) {
            return true;
        } else {
            return false;
        }
    }
}
```

> Main类

*Main*类首先生成了*Alice*至*Fred*等6个解决问题的实例。虽然此处定义的变量都是*Support*类型的，但是实际上所保存的变量却是*NoSupport、LimitSupport、SpecialSupport、OddSupport*等各个类的实例。

接下来，*Main*类调用*setNext*方法将*Alice*至*Fred*这6个实例串联在指责链上。之后，*Main*类逐个生成问题，并将它们传递给*alice*，然后显示最终谁解决了该问题。

```java
public class Main {
    public static void main(String[] args) {
        Support alice = new NoSupport("Alice");
        Support bob = new LimitSupport("Bob", 100);
        Support charlie = new SpecialSupport("Charlie", 429);
        Support diana = new LimitSupport("Diana", 200);
        Support elmo = new OddSupport("Elmo");
        Support fred = new LimitSupport("Fred", 300);
        // 形成职责链
        alice.setNext(bob).setNext(charlie).setNext(diana).setNext(elmo).setNext(fred);
        // 制造各种问题
        for (int i=0; i < 500; i+=33) {
            alice.support(new Trouble(i));
   		}
    }
}
```

**运行结果**

<img src="F:\文档\Typora Files\markdown-notes\images\notes\设计模式\chain_of_responsibility.png" alt="chain_of_responsibility" style="zoom:80%;" />

### 14.3 Chain of Responsibility模式中登场的角色

+ ***Handler*（处理者）**

*Handler*角色定义了处理请求的接口（API）。*Handler*角色知道“下一个处理者”是谁，如果自己无法处理请求，它会将请求转给“下一个处理者”。当然，“下一个处理者”也是*Handler*角色。在示例程序中，由*Support*类扮演此角色。负责处理请求的是*support*方法。

+ ***ConcreteHandle*（具体的处理者）**

*ConcreteVisitor*角色是处理请求的具体角色。在示例程序中，由*NoSupport、LimitSupport、OddSupport、SpecialSupport*等各个类扮演此角色。

+ ***Client*（请求者）**

*Client*角色是向第一个*ConcreteHandler*角色发送请求的角色。在示例程序中，由*Main*类扮演此角色。

### 14.4 拓展思路的要点

> 弱化了发出请求的人和处理请求的人之间的关系

*Chain of Responsibility*模式的最大优点就在于它弱化了发出请求的人（*Client*角色）和处理请求的人（*ConcreteHandler*角色）之间的关系。Client角色向第一个*ConcreteHandler*角色发出请求，然后请求会在职责链中传播，直到某个*ConcreteHandler*角色处理该请求。

如果不使用该模式，就必须有某个伟大的角色知道“谁应该处理什么请求”，这有点类似中央集权制。而让“发出请求的人”知道“谁应该处理该请求”并不明智，因为如果发出请求的人不得不知道处理请求的人各自的责任分担情况，就会降低其作为可复用的组件的独立性。

【补充说明】为了简单起见，在示例程序中，我们让扮演*Client*角色的*Main*类负责串联起*ConcreteHandler*的职责链。

> 可以动态地改变职责链

在示例程序中，问题的解决是按照从*Alice*到*Fred*的固定顺序进行处理的。但是我们还需要考虑负责处理的各个*ConcreteHandler*角色之间的关系可能会发生变化的情况。如果使用*Chain of Responsibility*模式，通过委托推卸责任，就可以根据情况变化动态地重组职责链。

如果不使用*Chain of Responsibility*模式，而是在程序中固定写明“某个请求需要谁处理”这样的对应关系，那么很难在程序运行中去改变请求的处理者。

在视窗系统中，用户有时需要可以自由地在视窗中添加控件（按钮和文本输入框等）。这时，*Chain of Responsibility*模式就有了用武之地。

> 专注于自己的工作

> 推卸请求会导致处理延迟

### 14.5 相关的设计模式

+ ***Composite*模式**

*Handler*角色经常会使用*Composite*模式。

+ ***Command*模式**

有时会使用*Command*模式向*Handler*角色发送请求。