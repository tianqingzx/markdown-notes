@[toc]

### 4.1 Factory Method 模式

**用*Template Method*模式来构建生成实例的工厂，这就是*Factory Method*模式。**

在*Factory Method*模式中，父类决定实例的生成方式，但并不决定所要生成的具体的类，具体的的处理全部交给子类负责。这样就可以生成实例的框架（framework）和实际负责生成实例的类解耦。

### 4.2 示例程序

该示例程序的作用是制作身份证（ID卡），它其中有5个类。

`Product`类和`Factory`类属于*framework*包。这两个类组成了生成实例的框架。

`IDCard`类和`IDCardFactory`类负责实际的加工处理，它们属于*idcard*包。

`Main`类是用于测试程序行为的类。

+ **生成实例的框架（framework包）**
+ **加工处理（idcard包）**

**类的一览表**

| 包        | 名字          | 说明                                           |
| --------- | ------------- | ---------------------------------------------- |
| framework | Product       | 只定义抽象方法 use 的抽象类                    |
| framework | Factory       | 实现了 create 方法的抽象类                     |
| idcard    | IDCard        | 实现了 use 方法的类                            |
| idcard    | IDCardFactory | 实现了 createProduct、registerProduct 方法的类 |
| 无名      | Main          | 测试程序行为的类                               |

**示例程序的类图**

<img src="F:\文档\Typora Files\markdown-notes\images\notes\设计模式\factory_method_uml.png" alt="factory_method_uml" style="zoom:80%;" />

> Product 类

```java
package framework;

public abstract class Product {
    public abstract void use();
}
```

> Factory 类

```java
package framework;

public abstract class Factory {
    public final Product create(String owner) {
        Product p = createProduct(owner);
        registerProduct(p);
        return p;
    }
    protected abstract Product createProduct(String owner);
    protected abstract void registerProduct(Product product);
}
```

> IDCard 类

```java
package idcard;
import framework.*;

public class IDCard extends Product {
    private String owner;
    IDCard(String owner) {
        System.out.println("制作" + owner + "的ID卡。");
        this.owner = owner;
    }
    @Override
    public void use() {
        System.out.println("使用" + owner + "的ID卡。");
    }
    public String getOwner() {
        return owner;
    }
}
```

> IDCardFactory 类

```java
package idcard;
import framework.*;

public class IDCardFactory extends Factory {
    private List<String> owners = new ArrayList<>();
    @Override
    protected Product createProduct(String owner) {
        return new IDCard(owner);
    }
    @Override
    protected void registerProduct(Product product) {
        owners.add(((IDCard)product).getOwner());
    }
    public List<String> getOwners() {
        return owners;
    }
}
```

> Main 类

```java
import framework.*;
import idcard.*;

public class Main {
    public static void main(String[] args) {
        Factory factory = new IDCardFactory();
        Product card1 = factory.create("小明");
        Product card2 = factory.create("小红");
        Product card3 = factory.create("小刚");
        card1.use();
        card2.use();
        card3.use();
    }
}
```

**运行结果**

<img src="F:\文档\Typora Files\markdown-notes\images\notes\设计模式\factory_method.png" alt="factory_method" style="zoom:80%;" />

### 4.3 Factory Method 模式中的登场角色

父类（框架）这一方的*Creator*角色和*Product*角色的关系与子类（具体加工）这一方的*ConcreteCreator*角色和*ConcreteProduct*角色的关系是平行的。

**Factory Method 模式的类图**

<img src="F:\文档\Typora Files\markdown-notes\images\notes\设计模式\factory_method_uml_2.png" alt="factory_method_uml_2" style="zoom:80%;" />

+ **Product（产品）**

*Product*角色属于框架这一方，是一个抽象类。它定义了在*Factory Method*模式中生成的那些实例所持有的接口（API），但具体的处理则由子类*ConcreteProduct*角色决定。在示例程序中，由*Product*类扮演此角色。

+ **Creator（创建者）**

*Creator*角色属于框架这一方，它是负责生成*Product*角色的抽象类，但具体的处理则由子类*ConcreteCreator*角色决定。在示例程序中，由*Factory*类扮演此角色。

*Creator*角色对于实际负责生成实例的*ConcreteCreator*角色一无所知，它唯一知道的就是，只要调用*Product*角色和生成实例的方法，就可以生成*Product*的实例。在示例程序中，*createProduct*方法是用于生成实例的方法。**不用 new 关键字来生成实例，而是调用生成实例的专用方法来生成实例，这样就可以防止父类与其它具体类耦合。**

+ **ConcreteProduct（具体的产品）**

*ConcreteProduct*角色属于具体加工这一方，它决定了具体的产品。在示例程序中，由*IDCard*类扮演此角色。

+  **ConcreteCreator（具体的创建者）**

*ConcreteCreator*角色属于具体加工这一方，它负责生成具体的产品。在示例程序中，由*IDCardFactory*类扮演此角色。

### 4.4 扩展思路的要点

##### 框架与具体加工

我们不需要修改*framework*包中的任何内容，就可以创建出其它的“产品”和“工厂”。

例如，我们要创建表示电视机的类*Televsion*和表示电视机工厂的类*TelevsionFactory*。这时，我们只需要引入*(import) framework*包就可以编写*televsion*包。

##### 生成实例——方法的三种实现方式

*Factory*类的*createProduct*方法是抽象方法，也就是说需要在子类中实现该方法。*createProduct*方法的实现方式一般有以下3种。

+ 指定其为抽象方法

```java
abstract class Factory {
    public abstract Product createProduct(String name);
    ...
}
```

+ 为其实现默认处理

```java
class Factory {
    public Product createProduct(String name) {
        return new Product(name);
    }
    ...
}
```

+ 在其中抛出异常

会在子类未实现该方法时，就会在运行时报错，告知没有实现*createProduct*方法。

```java
class Factory {
    public Product createProduct(String name) {
        throw new FactoryMethodRuntimeException();
    }
    ...
}
```

不过，需要另外编写*FactoryMethodRuntimeException*异常类。

### 4.5 相关的设计模式

+ **Template Method 模式** 

*Factory Method*模式是*Template Method*的典型应用。在示例程序中，*create*方法就是模板方法。

+ **Singleton 模式**

在多数情况下我们都可以将*Singleton*模式用于扮演*Creator*角色（或者*ConcreteCreator*角色）的类。这是因为在程序中没有必要存在多个*Creator*角色（或是*ConcreteCreator*角色）的实例。不过在示例程序中，我们并没有使用*Singleton*模式。

+ **Composite 模式**

有时可以将*Composite*模式用于*Product*角色（或是*ConcreteProduct*角色）。

+ **Iterator 模式**
+ 有时，在*Iterator*模式中使用*iterator*方法生成*Iterator*的实例时会使用*Factory Method*模式。