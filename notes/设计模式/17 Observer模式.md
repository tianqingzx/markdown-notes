# 第17章 Observer模式

### 发送状态变化通知



### 17.1 Observer模式

*Observer*的意思是“进行观察的人”，也就是“观察者”的意思。

在*Observer*模式中，当观察对象的状态发生变化时，会通知给观察者。*Observer*模式适用于根据对象状态进行相应处理的场景。



### 17.2 示例程序

这是一段简单的示例程序，观察者将观察一个会生成数值的对象，并将它生成的数值结果显示出来。不过，不同的观察者的显示方式不一样。*DigitObserver*会以数字形式显示数值，而*GraphObserver*则会以简单的图示形式来显示数值。

**类和接口一览表**

| 名字                  | 说明                             |
| --------------------- | -------------------------------- |
| Observer              | 表示观察者的接口                 |
| NumberGenerator       | 表示生成数值的对象的抽象类       |
| RandomNumberGenerator | 生成随机数的类                   |
| DigitObserver         | 表示以数字形式显示数值的类       |
| GraphObserver         | 表示以简单的图示形式显示数值的类 |
| Main                  | 测试程序行为的类                 |

**示例程序类图**

![observer_uml](F:\文档\Typora Files\markdown-notes\images\notes\设计模式\observer_uml.PNG)

> Observer接口

*Observer*接口是表示“观察者”的接口。具体的观察者会实现这个接口。

需要注意的是，这个*Observer*接口是为了便于我们了解*Observer*的示例程序而编写的，它与*Java*类库中的*java.util.Observer*接口不同。

用于生成数值的*NumberGenerator*类会调用*update*方法。*Generator*有“生成器”“产生器”的意思。如果调用*update*方法，*NumberGenerator*类就会将“生成的数值发生了变化，请更新显示内容”的通知发送给*Observer*。

```java
public interface Observer {
    public abstract void update(NumberGenerator generator);
}
```



> NumberGenerator类

*NumberGenetator*类是用于生成数值的抽象类。生成数值的方法（*execute*方法）和获取数值的方法（*getNumber*方法）都是抽象方法，需要子类去实现。

*observers*字段中保存有观察*NumberGenerator*的*Observer*们。

*addObserver*方法用于注册*Observer*，而*deleteObserver*方法用于删除*Observer*。

*notifyObservers*方法会向所有的*Observer*发送通知，告诉它们“我生成的数值发生了变化，请更新显示内容”。该方法会调用每个*Observer*的*update*方法。

```java
import java.util.ArrayList;
import java.util.Iterator;

public abstract class NumberGenerator {
    private ArrayList observers = new ArrayList();  // 保存Observer们
    public void addObserver(Observer observer) {    // 注册Observer
        observers.add(observer);
    }
    public void deleteObserver(Observer observer) { // 删除Observer
        observers.remove(observer);
    }
    public void notifyObservers() {                 // 向Observer发送通知
        Iterator it = observers.iterator();
        while (it.hasNext()) {
            Observer o = (Observer) it.next();
            o.update(this);
        }
    }
    public abstract int getNumber();                // 获取数值
    public abstract void execute();                 // 生成数值
}
```



> RandomNumberGenerator类

*RandomNumberGenerator*类是*NumberGenerator*的子类，它会生成随机数。

*random*字段中保存有*java.util.Random*类的实例（即随机数生成器）。而*number*字段中保存的是当前生成的随机数。

*getNumber*方法用于获取*number*字段的值。

*execute*方法会生成20个随机数（0～49的整数），并通过*notifyObservers*方法把每次生成结果通知给观察者。这里使用的*nextInt*方法是*java.util.Random*类的方法，它的功能是返回下一个随机整数值（取值范围大于0，小于指定值）。

```java
import java.util.Random;

public class RandomNumberGenerator extends NumberGenerator {
    private Random random = new Random();   // 随机生成器
    private int number;                     // 当前数值
    @Override
    public int getNumber() {                // 获取当前数值
        return number;
    }
    @Override
    public void execute() {
        for (int i = 0; i < 20; i++) {
            number = random.nextInt(50);
            notifyObservers();
        }
    }
}
```



> DigitObserver类

*DigitObserver*类实现了*Observer*接口，它的功能是以数字形式显示观察到的数值。它的*update*方法接收*NumberGenerator*的实例作为参数，然后通过调用*NumberGenerator*类的实例的*getNumber*方法可以获取到当前的数值，并将这个数值显示出来。为了能够让大家看清它是如何显示数值的，这里我们使用*Thread.sleep*来降低了程序的运行速度。

```java
public class DigitObserver implements Observer {
    @Override
    public void update(NumberGenerator generator) {
        System.out.println("DigitObserver:" + generator.getNumber());
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            System.out.println(e.toString());
        }
    }
}
```



> GraphObserver类

*GraphObserver*类也实现了*Observer*接口。该类会将观察到的数值以\*\*\*\*\*\*\*这样的简单图示的形式显示出来。

```java
public class GraphObserver implements Observer {
    @Override
    public void update(NumberGenerator generator) {
        System.out.print("GraphObserver:");
        int count = generator.getNumber();
        for (int i = 0; i < count; i++) {
            System.out.print("*");
        }
        System.out.println("");
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            System.out.println(e.toString());
        }
    }
}
```



> Main类

*Main*类生成了一个*RandomNumberGenerator*类的实例和两个观察者，其中*observer1*是*DigitObserver*类的实例，*observer2*是*GraphObserver*类的实例。

在使用*addObserver*注册观察者后，它还会调用*generator.execute*方法生成随机数值。

```java
public class Main {
    public static void main(String[] args) {
        NumberGenerator generator = new RandomNumberGenerator();
        Observer observer1 = new DigitObserver();
        Observer observer2 = new GraphObserver();
        generator.addObserver(observer1);
        generator.addObserver(observer2);
        generator.execute();
    }
}
```

<img src="F:\文档\Typora Files\markdown-notes\images\notes\设计模式\observer.PNG" alt="observer" style="zoom:80%;" />

### 17.3 Observer模式中的登场角色

+ ***Subject*（观察对象）**

*Subject*角色表示观察对象。*Subject*角色定义了注册观察者和删除观察者的方法。此外，它还声明了“获取现在的状态”的方法。在示例程序中，由*NumberGenerator*类扮演此角色。

+ ***ConcreteSubject*（具体的观察对象）**

*ConcreteSubject*角色表示具体的被观察对象。当自身状态发生变化后，它会通知所有已经注册的*Observer*角色。在示例程序中，由*RandomNumberGenerator*类扮演此角色。

+ ***Observer*（观察者）**

*Observer*角色负责接收来自*Subject*角色的状态变化的通知。为此，它声明了*update*方法。在示例程序中，由*Observer*接口扮演此角色。

+ ***ConcreteObserver*（具体的观察者）**

*ConcreteObserver*角色表示具体的*Observer*。当它的*update*方法被调用后，会去获取要观察的对象的最新状态。在示例程序中，由*DigitObserver*类和*GraphObserver*类扮演此角色。



### 17.4 拓展思路的要点

> 这里也出现了可替换性

使用设计模式的目的之一就是使类成为可复用的组件。

在*Observer*模式中，有带状态的*ConcreteSubject*角色和接收状态变化通知的*ConcreteObserver*角色。连接这两个角色的就是它们的接口*（API）Subject*角色和*Observer*角色。

一方面*RandomNumberGenerator*类并不知道，也无需在意正在观察自己的（自己需要通知的对象）到底是*DigitObserver*类的实例还是*GraphObserver*类的实例。不过它知道在它的*observers*字段中所保存的观察者们都实现了*Observer*接口。因为这些实例都是通过*addObserver*方法注册的，这就确保了它们一定都实现了*Observer*接口，一定可以调用它们的*update*方法。

另一方面，*DigitObserver*类也无需在意自己正在观察的究竟是*RandomNumberGenerator*类的实例还是其他*XXXXNumberGenerator*类的实例。不过，*DigitObserver*类知道它们是*NumberGenerator*类的子类的实例，并持有*getNumber*方法。

+ **利用抽象类和接口从具体类中抽出抽象方法**

+ **在将实例作为参数传递至类中，或者在类的字段中保存实例时，不使用具体类型，而是使用抽象类型和接口**

这样的实现方式可以帮助我们轻松替换具体类。



> Observer的顺序

*Subject*角色中注册有多个*Observer*角色。在示例程序的*notifyObservers*方法中，先注册的*Observer*的*update*方法会先被调用。

通常，在设计*ConcreteObserver*角色的类时，需要注意这些*Observer*的*update*方法的调用顺序，不能因为*update*方法的调用顺序发生改变而产生问题。例如，在示例程序中，绝不能因为先调用*DigitObserver*的*update*方法后调用*GraphObserver*的*update*方法而导致应用程序不能正常工作。当然，通常，只要保持各个类的独立性，就不会发生上面这种类的依赖关系混乱的问题。



> 当Observer的行为会对Subject产生影响时

在本节的示例程序中，*RandomNumberGenerator*类会在自身内部生成数值，调用*update*方法。不过，在通常的*Observer*模式中，也就可能是其他类触发*Subject*角色调用*update*方法。例如，在*GUI*应用程序中，多数情况下是用户按下按钮后会触发*update*方法被调用。

当然，*Observer*角色也有可能会触发*Subject*角色调用*update*方法。这时，如果稍不留神，就可能会导致方法被循环调用。

Subject状态发生变化—>通知Observer—>Observer调用Subject的方法—>导致Subject状态发生变化—>通知Observer—>......



> 传递更新信息的方式

*NumberGenerator*利用*update*方法告诉*Observer*自己的状态发生了更新。传递给*update*方法的参数只有一个，就是调用*update*方法的*NumberGenerator*的实例自身。*Observer*会在*update*方法中调用该实例的*getNumber*来获取足够的数据。

不过在示例程序中，*update*方法接收到的参数中并没有被更新的数值。也就是说，*update*方法的定义可能不是如下（1）中这样，而是如下（2）中这样，或者更简单的（3）这样的。

```java
void update(NumberGenerator generator);              ......(1)
void update(NumberGenerator generator, int number);  ......(2)
void update(int number);                             ......(3)
```

（1）只传递了*Subject*角色作为参数。*Observer*角色可以从*Subject*角色中获取数据。

（2）除了传递*Subject*角色以外，还传递了*Observer*所需的**数据**（这里指的是所有的更新信息）。这样就省去了*Observer*自己获取数据的麻烦。不过，这样做的话，*Subject*角色就知道了*Observer*所要进行的处理的内容了。

在很复杂的程序中，让*Subject*角色知道*Observer*角色所要进行的处理会让程序变得缺少灵活性。例如，假设现在我们需要传递上次传递的数值和当前的数值之间的差值，那么我们就必须在*Subject*角色中先计算出这个差值。因此，我们需要综合考虑程序的复杂度来设计*update*方法的参数的最优方案。

（3）比（2）简单，省略了*Subject*角色。示例程序同样也适用这种实现方式。不过，如果一个*Observer*角色需要观察多个*Subject*角色的时候，此方式就不适用了。这是因为*Observer*角色不知道传递给*update*方法的参数究竟是其中哪个*Subject*角色的数值。



> 从“观察”变为“通知”

*Observer*本来的意思是“观察者”，但实际上*Observer*角色并非主动地去观察，而是被动地接受来自*Subject*角色的通知。因此，*Observer*模式也被称为*Publish-Subscribe*（发布—订阅）模式。



> Model/View/Controller(MVC)

*MVC*中的*Model*和*View*的关系与*Subject*角色和*Observer*角色的关系相对应。*Model*是指操作“不依赖于显示形式的内部模型”的部分，*View*则是管理*Model*“怎样显示”的部分。通常情况下，一个*Model*对应多个*View*。



### 17.5 延伸阅读：java.util.Observer接口

*Java*类库中的*java.util.Observer*接口和*java.util.Observable*类就是一种*Observer*模式。

*java.util.Observer*接口中定义了以下方法。



而*update*方法的参数则接收到了如下内容。

+ ***Observable*类的实例是被观察的*Subject*角色**

+ ***Object*类的实例是附加信息**

这与上文中提到的类型（2）相似。

但是*java.util.Observer*接口和*java.util.Observable*类并不好用。理由很简单，传递给*java.util.Observer*接口的*Subject*角色必须是*java.util.Observable*类型（或者它的子类型）的。但*Java*只能单一继承，也就说如果*Subject*角色已经是某个类的子类了，那么它将无法继承*java.util.Observable*类。



### 17.6 相关的设计模式

+ ***Mediator*模式**

在*Mediator*模式中，有时会使用*Observer*模式来实现*Mediator*角色与*Colleague*角色之间的通信。

就“发送状态变化通知”这一点而言，*Mediator*模式与*Observer*模式是类似的。不过，两种模式中，通知的目的和视角不同。

在*Mediator*模式中，虽然也会发送通知，不过那不过是为了对*Colleague*角色进行仲裁而已。

而在*Observer*模式中，将*Subject*角色的状态变化通知给*Observer*角色的目的则主要是为了使*Subject*角色和*Observer*角色同步。