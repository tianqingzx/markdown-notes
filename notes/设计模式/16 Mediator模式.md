# 第16章 Mediator模式
### 只有一个仲裁者

### 16.1 Mediator模式

可以理解为**组员向仲裁者报告，仲裁者向组员下达指示**。组员之间不再相互询问和相互指示。

*Mediator*的意思是“仲裁者”“中介者”。一方面，当发生麻烦事情的时候，通知仲裁者；当发生涉及全体组员的事情时，也通知仲裁者。当仲裁者下达指示时，组员会立即执行。团队组员之间不再相互沟通并私自做出决定，而是发生任何事情都向仲裁者报告。另一方面，仲裁者站在整个团队的角度上对组员上报的事情做出决定。这就是Mediator模式。

在*Mediator*模式中，“仲裁者”被称为Mediator，各组员被称为*Colleague*。

### 16.2 示例程序

这段示例程序是一个*GUI*应用程序，它展示了一个登陆对话框，用户在其中输入正确的用户名和密码后可以登录。

对话框的使用方法如下。

+ 可以选择作为游客访问（*Guest*）或是作为用户登录（*Login*）

+ 作为用户登录时，需要输入正确的用户名（*Username*）和密码（*Password*）

+ 点击*OK*按钮可以登录，点击*Cancel*按钮可以取消登录

  （在示例程序中我们不会真正登录，而是在按下按钮后就退出程序）

+ 如果选择作为游客访问，那么禁用用户名输入框和密码输入框，使用户无法输入

+ 如果选择作为用户登录，那么启用用户名输入框和密码输入框，使用户可以输入

+ 如果在用户名输入框中一个字符都没有输入，那么禁用密码输入框，使用户无法输入密码

+ 如果在用户名输入框中输入了至少一个字符，那么启用密码输入框，使用户可以输入密码（当然，如果选择作为游客访问，那么密码框依然是禁用状态）

+ 只有当用户名输入框和密码输入框中都至少输入一个字符后，OK按钮才处于启用状态，可以被按下。用户名输入框或密码输入框中一个字符都没有被输入的时候，禁用OK按钮，使其不可被按下（当然，如果选择作为游客访问，那么OK按钮总是处于启用状态）

+ *Cancel*按钮总是处于启用状态，任何时候都可以按下该按钮

像上面这样**要调整多个对象之间的关系时，就需要用到*Mediator*模式了**。即不让各个对象之间互相通信，而是增加一个仲裁者角色，让它们各自与仲裁者通信。然后，**将控制显示的逻辑处理交给仲裁者负责。**

**类和接口的一览表**

| 名字               | 说明                                                    |
| ------------------ | ------------------------------------------------------- |
| Mediator           | 定义“仲裁者”的接口（API）的接口                         |
| Colleague          | 定义“组员”的接口（API）的接口                           |
| ColleagueButton    | 表示按钮的类。它实现了Colleague接口                     |
| ColleagueTextField | 表示文本输入框的类。它实现了Colleague接口               |
| ColleagueCheckbox  | 表示勾选框（此处是单选按钮）的类。它实现了Colleague接口 |
| LoginFrame         | 表示登录对话框的类。它实现了Mediator接口                |
| Main               | 测试程序行为的类                                        |

**示例程序的类图**

![mediator_uml](F:\文档\Typora Files\markdown-notes\images\notes\设计模式\mediator_uml.png)

> Mediator接口

*Mediator*接口是表示仲裁者的接口。具体的仲裁者（后文中即将学习的*LoginFrame*类）会实现这个接口。

*createColleagues*方法用于生成Mediator要管理的组员。在示例程序中，*createColleagues*会生成对话框中的按钮和文本输入框等控件。

*colleagueChanged*方法会被各个*Colleague*组员调用。它的作用是让组员可以向仲裁者进行报告。在本例中，当单选按钮和文本输入框的状态发生变化时，该方法会被调用。

```java
public interface Mediator {
    public abstract void createColleagues();
    public abstract void colleagueChanged();
}
```

> Colleague接口

*Colleague*接口是表示向仲裁者进行报告的组员的接口。具体的组员（*ColleagueButton、ColleagueTextField、ColleagueCheckbox*）会实现这个接口。

*LoginFrame*类实现了*Mediator*接口，它首先会调用*setMediator*方法。该方法的作用是告知组员“我是仲裁者，有事请报告我”。向该方法中传递的参数是仲裁者的实例，之后在需要向仲裁者报告时（即调用*colleagueChanged*方法时）会用到该实例。

*setColleagueEnabled*方法的作用是告知组员仲裁者所下达的指示。参数*enabled*如果为*true*，就表示自己需要变为“启用状态”；如果是*false*，则表示自己需要变为“禁用状态”。这个方法表明，究竟是变为“启用状态”还是变为“禁用状态”，并非由组员自己决定，而是由仲裁者来决定。

此外需要说明的是，关于*Mediator*接口和*Colleague*接口中究竟需要定义哪些方法这点，是根据需求不同而不同的。在示例程序中，我们在*Mediator*中定义了*colleagueChanged*方法，在*Colleague*接口中定义了*setColleagueEnabled*方法。如果需要让*Mediator*角色和*Colleague*角色之间进行更加详细的通信，还需要定义更多的方法。也就是说，即使两段程序都使用了*Mediator*模式，但它们实际定义的方法可能会不同。

```java
public interface Colleague {
    public abstract void setMediator(Mediator mediator);
    public abstract void setColleagueEnabled(boolean enabled);
}
```

> ColleagueButton类

*ColleagueButton*类是*java.awt.Button*的子类，它实现了*Colleague*接口，与*LoginFrame*（*Mediator*接口）共同工作。

*mediator*字段中保存了通过*setMediator*方法的参数传递进来的*Mediator*对象（*LoginFrame*类的实例）。*setColleagueEnabled*方法会调用*Java*的*GUI*中定义的*setEnabled*方法，设置禁用或是启用控件。*setEnabled（true）*后控件按钮可以被按下，*setEnabled(false)*后按钮无法被按下。

```java
public class ColleagueButton extends Button implements Colleague {
    private Mediator mediator;
    public ColleagueButton(String caption) {
        super(caption);
    }
    @Override
    public void setMediator(Mediator mediator) {  // 保存Mediator
        this.mediator = mediator;
    }
    @Override
    public void setColleagueEnabled(boolean enabled) {  // Mediator下达启用/禁用的指示
        setEnabled(enabled);
    }
}
```

> ColleagueTextField类

*ColleagueTextField*类是*java.awt.TextField*的子类，它不仅实现了*Colleague*接口，还实现了*java.awt.event.TextListener*接口。这是因为我们希望通过*textValueChanged*方法捕捉到文本内容发生变化这一事件，并通知仲裁者。

在*Java*语言中，我们虽然无法继承（*extends*）多个类，但是我们可以实现（*implements*）多个接口。在*setColleagueEnabled*方法中，我们不仅调用了*setEnabled*方法，还调用了*setBackground*方法。这是因为我们希望在启用控件后，将它的背景色改为白色；禁用控件后，将它的背景色改为灰色。

*textValueChanged*方法是在*TextListener*接口中定义的方法。当文本内容发生变化时，*AWT*框架会调用该方法。在示例程序中，*textValueChanged*方法调用了*colleagueChanged*方法，这是在向仲裁者表达“对不起，文本内容有变化，请处理。”的意思。

```java
public class ColleagueTextField extends TextField implements TextListener, Colleague {
    private Mediator mediator;
    public ColleagueTextField(String text, int columns) {  // 构造函数
        super(text, columns);
    }
    @Override
    public void setMediator(Mediator mediator) {  // 保存Mediator
        this.mediator = mediator;
    }
    @Override
    public void setColleagueEnabled(boolean enabled) {  // Mediator下达启用/禁用的指示
        setEnabled(enabled);
        setBackground(enabled ? Color.white : Color.lightGray);
    }
    @Override
    public void textValueChanged(TextEvent e) {  // 当文字发生变化时通知Mediator
        mediator.colleagueChanged();
    }
}
```

> ColleagueCheckbox类

*ColleagueCheckbox*类是*java.awt.Checkbox*的子类。在示例程序中，我们将其作为单选按钮使用，而没有将其作为勾选框使用（使用*CheckboxGroup*）。

该类实现了*java.awt.event.ItemListener*接口，这是因为我们希望通过*itemSateChanged*方法来捕获单选按钮的状态变化。

```java
public class ColleagueCheckbox extends Checkbox implements ItemListener, Colleague {
    private Mediator mediator;
    public ColleagueCheckbox(String caption, CheckboxGroup group, boolean state) {
        super(caption, group, state);
    }
    @Override
    public void setMediator(Mediator mediator) {
        this.mediator = mediator;
    }
    @Override
    public void setColleagueEnabled(boolean enabled) {
        setEnabled(enabled);
    }
    @Override
    public void itemStateChanged(ItemEvent e) {  // 当状态发生变化时通知Mediator
        mediator.colleagueChanged();
    }
}
```

> LoginFrame类

现在，我们终于可以看看仲裁者的代码了。*LoginFrame*类是*java.awt.Frame*（用于编写*GUI*程序的类）的子类，它实现了*Mediator*接口。

*LoginFrame*类的构造函数进行了以下处理。

+ **设置背景色**

+ **设置布局管理器（配置4（纵）X2（横）窗格）**

+ **调用*createColleagues*方法生成*Colleague***

+ **配置*Colleague***

+ **设置初始状态**

+ **显示**

*createColleagues*方法会生成登录对话框所需的*Colleague*，并将它们保存在*LoginFrame*类的字段中。此外，它还会调用每个*Colleague*的*setMediator*方法，事先告知它们“我是仲裁者，有什么问题的可以向我报告”。*createColleagues*方法还设置了各个*Colleague*的*Listener*。这样，*AWT*框架就可以调用合适的*Listener*了。

整个示例程序中最重要的方法当属*LoginFrame*类的*colleagueChanged*方法。该方法负责前面讲到过的“设置控件的启用？禁用的复杂逻辑处理”。请大家回忆一下之前学习过的*ColleagueButton、ColleagueCheckbox、ColleagueTextField*等各个类。这些类中虽然都有设置自身的启用/禁用状态的方法，但是并没有“具体什么情况下需要设置启用/禁用”的逻辑处理。它们都只是简单地调用仲裁者的*colleagueChanged*方法告知仲裁者“剩下的就拜托给你了”。也就是说，所有最终的决定都是由仲裁者的*colleagueChanged*方法下达的。

通过*getState*方法可以获取单选按钮的状态，通过*getText*方法可以获取文本输入框中的文字。那么剩下的工作就是在*colleagueChanged*方法中实现之前学习过的那段复杂的控制逻辑处理了。此外，这里我们提取了一个共同的方法*userpassChanged*。该方法仅在*LoginFrame*类内部使用，其可见性为*private*。

```java
public class LoginFrame extends Frame implements ActionListener, Mediator {
    private ColleagueCheckbox checkGuest;
    private ColleagueCheckbox checkLogin;
    private ColleagueTextField textUser;
    private ColleagueTextField textPass;
    private ColleagueButton buttonOK;
    private ColleagueButton buttonCancel;
    // 生成并配置各个Colleague后，显示对话框
    public LoginFrame(String title) {
        super(title);
        setBackground(Color.lightGray);
        // 使用布局管理器生成4x2窗格
        setLayout(new GridLayout(4, 2));
        // 生成各个Colleague
        createColleagues();
        // 配置
        add(checkGuest);
        add(checkLogin);
        add(new Label("Username:"));
        add(textUser);
        add(new Label("Password:"));
        add(textPass);
        add(buttonOK);
        add(buttonCancel);
        // 设置初始的启用/禁用状态
        colleagueChanged();
        // 显示
        pack();
        show();
    }
    // 生成各个Colleague
    @Override
    public void createColleagues() {
        // 生成
        CheckboxGroup g = new CheckboxGroup();
        checkGuest = new ColleagueCheckbox("Guest", g, true);
        checkLogin = new ColleagueCheckbox("Login", g, false);
        textUser = new ColleagueTextField("", 10);
        textPass = new ColleagueTextField("", 10);
        textPass.setEchoChar('*');
        buttonOK = new ColleagueButton("OK");
        buttonCancel = new ColleagueButton("Cancel");
        // 设置Mediator
        checkGuest.setMediator(this);
        checkLogin.setMediator(this);
        textUser.setMediator(this);
        textPass.setMediator(this);
        buttonOK.setMediator(this);
        buttonCancel.setMediator(this);
        // 设置Listener
        checkGuest.addItemListener(checkGuest);
        checkLogin.addItemListener(checkLogin);
        textUser.addTextListener(textUser);
        textPass.addTextListener(textPass);
        buttonOK.addActionListener(this);
        buttonCancel.addActionListener(this);
    }
    // 接收来自于Colleague的通知然后判断各Colleague的启用/禁用状态
    @Override
    public void colleagueChanged() {
        if (checkGuest.getState()) {
            textUser.setColleagueEnabled(false);
            textPass.setColleagueEnabled(false);
            buttonOK.setColleagueEnabled(true);
        } else {
            textUser.setColleagueEnabled(true);
            userPassChanged();
        }
    }
    private void userPassChanged() {
        if (textUser.getText().length() > 0) {
            textPass.setColleagueEnabled(true);
            if (textPass.getText().length() > 0) {
                buttonOK.setColleagueEnabled(true);
            } else {
                buttonOK.setColleagueEnabled(false);
            }
        } else {
            textPass.setColleagueEnabled(false);
            buttonOK.setColleagueEnabled(false);
        }
    }
    @Override
    public void actionPerformed(ActionEvent e) {
        System.out.println(e.toString());
        System.exit(0);
    }
}
```

> Main类

*Main*类生成了*LoginFrame*类的实例。虽然*Main*类的*main*方法结束了，但是*LoginFrame*类的实例还一直被保存在*AWT*框架中。

```java
public class Main {
    public static void main(String[] args) {
        new LoginFrame("Mediator Sample");
    }
}
```

### 16.3 Mediator模式中的登场角色

+ ***Mediator*（仲裁者、中介者）**

*Mediator*角色负责定义与*Colleague*角色进行通信和做出决定的接口（*API*）。在示例程序中，由*Mediator*接口扮演此角色。

+ ***ConcreteMediator*（具体的仲裁者、中介者）**

*ConcreteMediator*角色负责实现*Mediator*角色的接口（*API*），负责实际做出决定。在示例程序中，由*LoginFrame*类扮演此角色。

+ ***Colleague*（同事）**

*Colleague*角色负责定义与*Mediator*角色进行通信的接口（*API*）。在示例程序中，由*Colleague*接口扮演此角色。

+ ***ConcreteColleague*（具体的同事）**

*ConcreteColleague*角色负责实现*Colleague*角色的接口（*API*）。在示例程序中，由*ColleagueButton*类、*ColleagueTextField*类和*ColleagueCheckbox*类扮演此角色。

### 16.4 拓展思路的要点

>  当发生分散灾难时

示例程序中的*LoginFrame*类的*colleagueChanged*方法稍微有些复杂。如果发生需求变更，该方法中很容易发生Bug。不过这并不是什么问题。因为即使*colleagueChanged*方法中发生了Bug，由于**其他地方并没有控制控件的启用/禁用状态的逻辑处理**，因此只要调试该方法就能很容易地找出Bug的原因。

请试想一下，如果这段逻辑分散在*ColleagueButton*类、*ColleagueTextField*类和*ColleagueCheckbox*类中，那么无论是编写代码还是调试代码和修改代码，都会非常困难。

通常情况下，面向对象编程可以帮助我们分散处理，避免处理过于集中，也就是说可以“分而治之”。但是在本章中的示例程序中，把处理分散在各个类中是不明智的。如果只是将应当分散的处理分散在各个类中，但是没有将应当集中的处理集中起来，那么这些分散的类最终只会导致灾难。

> 通信线路的增加

> 哪些角色可以复用

*ConcreteColleague*角色可以复用，但*ConcreteMediator*角色很难复用。
例如，假如我们现在需要制作另外一个对话框。这时，我们可将扮演*ConcreteColleague*角色的*ColleagueButton*类、*ColleagueTextField*类和*ColleagueCheckbox*类用于新的对话框中。这是因为在*ConcreteColleague*角色中并没有任何依赖于特定对话框的代码。

在示例程序中，依赖于特定应用程序的部分都被封装在扮演*ConcreteMediator*角色的*LoginFrame*类中。依赖于特定应用程序就意味着难以复用。因此，*LoginFrame*类很难在其他对话框中被复用。

### 16.5 相关的设计模式

+ ***Facade*模式**

在*Mediator*模式中，*Mediator*角色与*Colleague*角色进行交互。

而在*Facade*模式中，*Facade*角色单方面地使用其他角色来对外提供高层接口（*API*）。因此，可以说*Mediator*模式是双向的，而*Facade*模式是单向的。

+ ***Observer*模式**

有时会使用*Observer*模式来实现*Mediator*角色与*Colleague*角色之间的通信。