# 第23章 Interpreter模式

### 语法规则也是类



### 23.1 Interpreter模式

在*Interpreter*模式中，程序要解决的问题会被用非常简单的“迷你语言”表述出来，即用“迷你语言”编写的“迷你程序”把具体的问题表述出来。迷你程序是无法单独工作的，我们还需要用*Java*语言编写一个负责“翻译”（*interpreter*）的程序。翻译程序会理解迷你语言，并解释和运行迷你程序。这段翻译程序也被称为**解释器**。这样，当需要解决的问题发生变化时，不需要修改*Java*语言程序，只需要修改迷你语言程序即可应对。

下面，我们用图示展示一下当问题发生变化时，需要哪个级别的代码。使用*Java*语言编程时，需要修改的代码如图23-1所示。虽然我们希望需要修改的代码尽量少，但是多多少少都必须修改*Java*代码。

但是，在使用*Interpreter*模式后，我们就无需修改*Java*程序，只需修改用迷你语言编写的迷你程序即可。



### 23.2 迷你语言

> 迷你语言的命令

在开始学习*Interpreter*模式的示例程序之前，我们先来了解一下本章中涉及的“迷你语言”。迷你语言的用途是控制无线玩具车。虽说是控制无线玩具车，其实能做的事情不过以下3种。

+ 前进1米（*go*）
+ 右转（*right*）
+ 左转（*left*）

以上就是可以向玩具车发送的命令。*go*是前进1米后停止的命令；*right*是原地向右转的命令；*left*是原地向左转的命令。在实例操作时，是不能完全没有偏差地原地转弯的。为了使问题简单化，我们这里并不会改变玩具车的位置，而是像将其放在旋转桌子上一样，让它转个方向。

如果只是这样，大家可能感觉没什么意思。所以，接下来我们再加入一个循环命令。

+ 重复（*repeat*）

以上命令组合起来就是可以控制无线玩具车的迷你语言了。我们会在本章使用迷你语言学习*Interpreter*模式。



> 迷你语言程序示例

下面我们来看一段用迷你语言编写的迷你程序。下面这条语句可以控制无线玩具车前进（之后停止）。

`program go end`

为了便于大家看出语句的开头和结尾，我们在语句前后分别加上了*program*和*end*关键字（稍后会学习迷你语言的语法）。



接下来是一段让无线玩具车先前进一米，接着让它右转两次再返回来的程序。

`program go right right go end`

再接下来的这段程序是让无线玩具车按照正方形路径行进。

`program go right go right go right go right end    ......(A)`



（A）程序的最后（即*end*之前）之所以加上了一个*right*，是因为当无线玩具车回到起点后，我们希望它的方向与出发时相同。在（A）程序中，重复出现了4次*go right*。这样，我们可以使用*repeat...end*语句来实现下面的（B）程序（为了能够编写出这段程序，我们需要定义迷你语言的语法）。

`program repeat 4 go right end end    ......(B)`

在（B）程序的最后出现了两个*end*，其中第一个（左边）*end*表示*repeat*的结束，第二个（右边）*end*表示*program*的结束。也就是说，程序结构如下。

```java
program			程序开始
    repaet			循环开始
    	4				循环的次数
    	go				前进
    	right			右转
    end				循环结束
end				程序结束
```

在大家的脑海中，车轮是不是已经骨碌骨碌转起来了呢？那么，我们再一起看看下面这段程序是如何操作无线玩具车的。

`porgram repeat 4 repeat 3 go right go left end right end end`

现在，玩具车会按照锯齿形状路线前进。这里有两个*repeat*，可能会让大家有些难以理解，不过按照下面这样分解一下就很容易理解了。

```java
program				程序开始
    repeat				循环开始（外侧）
    	4					循环的次数
    	repeat				循环开始（内侧）
    		3					循环的次数
    		go					前进
    		right				右转
    		go					前进
    		left				左转
    	end					循环结束（内侧）
    	right				右转
    end					循环结束（外侧）
end					程序结束
```

内侧的循环语句是*go right go left*，它是一条让无线玩具车“前进后右转，前进后左转”的命令。该命令会重复3次。这样，玩具车就会向右沿着锯齿形线路行进。接着，退至外侧循环看，玩具车会连续4次“沿着锯齿形线路行进一次后，右转一次”。这样，最终行进路线就变成了一个锯齿样的棱形。



> 迷你语言的语法

这里使用的描述方法是*BNF*的一个变种。*BNF*是*Backus Naur Form*或*Backus Normal Form*的略称，它经常被用于描述语法。



我们按照自上而下的顺序进行学习。

`<program> ::= program <command list>`

首先，我们定义了程序*\<program\>*，即“所谓*\<program\>*，是指*program*关键字后面跟着的命令列表*\<command list\>*”。*“::=”*的左边表示定义的名字，右边表示定义的内容。

`<command list> ::= <command>* end`

接着，我们定义了命令列表*\<command list\>*，即“所谓*\<command list\>*，是指重复0次以上*\<command\>*后，接着一个*end*关键字”。“\*”表示前面的内容**循环0次以上**。

`<command> ::= <repeat command> | <primitive command>`

现在，我们来定义*\<command\>*，即“所谓*\<command\>*，是指*\<repeat command\>*或者*\<primitive command\>*”。该定义中的“｜”表示“或”的意思。

`<repeat command> ::= repeat <number><command list>`

接下来，我们定义循环命令，即“所谓*\<repeat command\>*，是指*repeat*关键字后面跟着循环次数*\<number\>*和要循环的命令列表*\<command list\>*”。其中的命令列表*\<command list\>*之前已经定义过了，而在定义命令列表*\<command list\>*的时候使用了*\<command\>*，在定义*\<command\>*的时候又使用了*\<repeat command\>*，而在定义*\<repeat command\>*的时候又使用了*\<command list\>*。像这样，**在定义某个东西时，它自身又出现在了定义的内容中，我们称这种定义为递归定义**。稍后，我们会使用*Java*语言实现迷你语言的解释器，到时候会有相应的代码结构来解释递归定义。

`<primitive command> ::= go | right | left`

这是基本命令*\<primitive command\>*的定义，即“所谓*\<primitive command\>*，是指*go*或者*right*或者*left*”。

最后只剩下*\<number\>*了，要想定义出全部的*\<number\>*可能非常复杂，这里我们省略了它的定义。总之，把*\<number\>*看作是3、4和12345这样的自然数即可。

**注意**：严格说，这里使用的是*EBNF*。在*BNF*中，循环不是用*表示的，而是用递归定义来表示的。



> 终结符表达式与非终结符表达式

前面讲到的像*\<primitive command\>*这样的不会被进一步展开的表达式被称为“终结符表达式”（*Nonterminal Expression*）。我们知道，巴士和列车的终到站被称为终点站，这里的终结符就类似于终点站，它表示语法规则的终点。

与之相对的是，像*\<program\>*和*\<command\>*这样的需要被进一步展开的表达式被称为“非终结符表达式”。



### 23.3 示例程序

这段示例程序实现了一个迷你程序的语法解析器。

在之前学习迷你程序的相关内容时，我们分别学习了对迷你程序的各个语法部分。像这样将迷你程序当作普通字符分解，然后看看各个部分分别是什么结构的过程，就是语法解析。

例如有如下迷你程序。

`program repeat 4 go right end end`

将这段迷你程序推导成为图中那样的结构（**语法树**）的处理，就是语法解析。

本章中的示例程序只会实现至推导出语法树。实际地“运行”程序的部分，我们将会在习题中实现。

**类的一览**

| 名字                 | 说明                               |
| -------------------- | ---------------------------------- |
| Node                 | 表示语法树“节点”的类               |
| ProgramNode          | 对应<program>的类                  |
| CommandListNode      | 对应<command list>的类             |
| CommandNode          | 对应<command>的类                  |
| RepeatCommandNode    | 对应<repeat command>的类           |
| PrimitiveCommandNode | 对应<primitive command>的类        |
| Context              | 表示语法解析上下文的类             |
| ParseException       | 表示语法解析中可能会发生的异常的类 |
| Main                 | 测试程序行为的类                   |

**示例程序的类图**

![interpreter_uml](F:\文档\Typora Files\markdown-notes\images\notes\设计模式\interpreter_uml.PNG)

> Node 类

*Node*类是语法树中各个部分（节点）中的最顶层的类。在*Node*类中只声明了一个*parse*抽象方法，该方法用于“进行语法解析处理”。但*Node*类仅仅是声明该方法，具体怎么解析交由*Node*类的子类负责。*parse*方法接收到的参数*Context*是表示语法解析上下文的类，稍后我们将来学习*parse*方法。在*parse*的声明中，我们使用了*throws*关键字。它表示在语法解析过程中如果发生了错误，*parse*方法就会抛出*ParseException*异常。

如果只看*Node*类，我们还无法知道具体怎么进行语法解析，所以我们接着往下看。

```java
public abstract class Node {
    public abstract void parse(Context context) throws ParseException;
}
```



> ProgramNode类

下面展示的迷你语言的语法描述（*BNF*）来看看各个类的定义。首先，我们看看表示程序*\<program\>*的*ProgramNode*类。在*ProgramNode*类中定义了一个*Node*类型的*CommandListNode*字段，该字段用于保存*\<command list\>*对应的结构（节点）。

那么，*ProgramNode*的*parse*方法究竟进行了什么处理呢？通过查看迷你语言的*BNF*描述我们可以发现，*\<program\>*的定义中最开始会出现*program*这个单词。因此，我们用下面的语句跳过这个单词。

`context.skipToken("program")`

我们称语法解析时的处理单位为标记（*token*）。在迷你语言中，“标记”相当于“英文单词”。在一般的编程语言中，“+”和“==”等也是标记。更具体地说，词法分析（*lex*）是从文字中得到标记，而语法解析（*parse*）则是根据标记推导出语法树。

上面的*skipToken*方法可以跳过*program*这个标记。如果没有这个标记就会抛出*ParseException*异常。
继续查看*BNF*描述会发现，在*program*后面会跟着*\<command list\>*。这里，我们会生成*\<command list\>*对应的*CommandListNode*类的实例，然后调用该实例的*parse*方法。*ProgramNode*类的方法并不知道*\<command list\>*的内容。即在*ProgramNode*类中实现的内容，并没有超出下面的*BNF*所描述的范围。

`<program> ::= program <command list>`

*toString*方法用于生成表示该节点的字符串。在*Java*中，连接实例与字符串时会自动调用实例的*toString*方法，因此如下（1）与（2）是等价的。

`"[program " + commandListNode + "]";    ......(1)`

`"[program " + commandListNode.toString() + "]";    ......(2)`

*toString*方法的实现也与上面的*BNF*描述完全相符。

```java
// <program> ::= program <command list>
public class ProgramNode extends Node {
    private Node commandListNode;
    @Override
    public void parse(Context context) throws ParseException {
        context.skipToken("program");
        commandListNode = new CommandListNode();
        commandListNode.parse(context);
    }
    public String toString() {
        return "[program " + commandListNode + " ]";
    }
}
```



> CommandListNode类

下面我们来看看*CommandListNode*类。*\<command list\>*的*BNF*描述如下。

`<command list> ::= <command>* end`

即重复0次以上*\<command\>*，然后以*end*结束。为了能保存0次以上的*\<command\>*，我们定义了*java.util.ArrayList*类型的字段*list*，在该字段中保存与*\<command\>*对应的*CommandNode*类的实例。

*CommandListNode*类的*parse*方法是怎么实现的呢？首先，如果当前的标记*context.currentToken()*是*null*，表示后面没有任何标记（也就是已经解析至迷你程序的末尾）了。这时，*parse*方法会先设置*ParseException*异常中的消息为“缺少*end（Missing ‘end’）*”，然后抛出*ParseException*异常。

接下来，如果当前的标记是*end*，表示已经解析至*\<command list\>*的末尾。这时，*parse*方法会跳过*end*，然后*break*出*while*循环。

再接下来，如果当前的标记不是*end*，则表示当前标记是*\<command\>*。这时，*parse*方法会生成与*\<command\>*对应的*commandNode*的实例，并调用它的*parse*方法进行解析。然后，还会将*commandNode*的实例*add*至*list*字段中。

大家应该看出来了，这里的实现也没有超出*BNF*描述的范围。我们在编程时要尽量忠实于*BNF*描述，原封不动地将*BNF*描述转换为*Java*程序。这样做可以降低出现*Bug*的可能性。在编程中加入读取更深层次的节点的处理，但这样反而可能会引入意想不到的*Bug*。*Interpreter*规模本来就采用了迷你语言这样的间接处理，所以要一些小聪明来试图提高效率并不明智。

```java
import java.util.ArrayList;

// <command list> ::= <command>* end
public class CommandListNode extends Node {
    private ArrayList list = new ArrayList();
    @Override
    public void parse(Context context) throws ParseException {
        while (true) {
            if (context.currentToken() == null) {
                throw new ParseException("Missing 'end'");
            } else if (context.currentToken().equals("end")) {
                context.skipToken("end");
                break;
            } else {
                Node commandNode = new CommandNode();
                commandNode.parse(context);
                list.add(commandNode);
            }
        }
    }
    public String toString() {
        return list.toString();
    }
}
```



> CommandNode类

如果理解了前面学习的*ProgramNode*类和*CommandListNode*类，那么应该也可以很快地理解*CommandNode*类。*\<command\>*的*BNF*描述如下。

`<command> ::= <repeat command> | <primitive command>`

在代码中的*Node*类型的*node*字段中保存的是与*\<repeat command\>*对应的*RepeatCommandNode*类的实例，或与*\<primitive command\>*对应的*PrimitiveCommandNode*类的实例。

```java
// <command> ::= <repeat command> | <primitive command>
public class CommandNode extends Node {
    private Node node;
    @Override
    public void parse(Context context) throws ParseException {
        if (context.currentToken().equals("repeat")) {
            node = new RepeatCommandNode();
            node.parse(context);
        } else {
            node = new PrimitiveCommandNode();
            node.parse(context);
        }
    }
    public String toString() {
        return node.toString();
    }
}
```



> RepeatCommandNode类

*RepeatCommandNode*类对应*\<repeat command\>*的类。*\<repeat command\>*的*BNF*描述如下。

`<repeat command> ::= repeat <number><command list>`

在代码中，*\<number\>*被保存在*int*型字段*number*中，*\<command list\>*被保存在*Node*型字段*CommandListNode*中。

现在，大家应该都注意到*parse*方法的递归关系了。让我们追溯一下*parse*方法的调用关系。

+ 在*RepeatCommandNode*类的*parse*方法中，会生成*CommandListNode*的实例，然后调用它的*parse*方法

+ 在*CommandListNode*的*parse*方法中，会生成*CommandNode*的实例，然后调用它的*parse*方法。

+ 在*CommandNode*类的*parse*方法中，会生成*RepeatCommandNode*的实例，然后调用它的*parse*方法

+ 在*RepeatCommandNode*类的*parse*方法中......

这样的*parse*方法调用到底要持续到什么时候呢？其实，它的终点就是终结符表达式。在*CommandNode*类的*parse*方法中，程序并不会一直进入*if*语句的*RepeatCommandNode*处理分支中，最终总是会进入*PrimitiveCommandNode*的处理分支。并且，不会从*PrimitiveCommandNode*的parse方法中再调用其他类的*parse*方法。

如果不习惯递归定义的处理方式，可能会感觉到这里似乎进入了死循环。其实这是错觉。不论是在*BNF*描述中还是在*Java*程序中，一定都会结束于终结符表达式。如果没有结束于终结符表达式，那么一定是语法描述有问题。

```java
// <repeat command> ::= repeat <number> <command list>
public class RepeatCommandNode extends Node {
    private int number;
    private Node commandListNode;
    @Override
    public void parse(Context context) throws ParseException {
        context.skipToken("repeat");
        number = context.currentNumber();
        context.nextToken();
        commandListNode = new CommandListNode();
        commandListNode.parse(context);
    }
    public String toString() {
        return "[repeat " + number + " " + commandListNode + "]";
    }
}
```



> PrimitiveCommandNode类

*PrimitiveCommandNode*类对应的*BNF*描述如下。

`<primitive command> ::= go | right | left`

确实，*PrimitiveCommandNode*类的*parse*方法没有调用其他类的*parse*方法。

```java
// <primitive command> ::= go | right | left
public class PrimitiveCommandNode extends Node {
    private String name;
    @Override
    public void parse(Context context) throws ParseException {
        name = context.currentToken();
        context.skipToken(name);
        if (!name.equals("go") && !name.equals("right") && !name.equals("left")) {
            throw new ParseException(name + " is undefined");
        }
    }
    public String toString() {
        return name;
    }
}
```



> Context类

至此，关于*Node*类以及它的子类的学习就全部结束了。剩下的就是*Context*类了。*Context*类提供了语法解析所必须的方法。

**Context类提供的方法**

| 名字          | 说明                                                   |
| ------------- | ------------------------------------------------------ |
| NextToken     | 获取下一个标记（前进至下一个标记）                     |
| currentToken  | 获取当前的标记（不会前进至下一个标记）                 |
| skipToken     | 先检查当前标记，然后获取下一个标记（前进至下一个标记） |
| currentNumber | 获取当前标记对应的数值（不会前进至下一个标记）         |

这里，我们使用*java.util.StringTokenizer*类来简化了我们的程序，它会将接收到的字符串分割为标记。在分割字符串时使用的分隔符是空格“‘’”、制表符“‘\t’”、换行符“‘\n’”、回车符“‘\r’”、换页符“‘\f’”（也可以使用其他分隔符，请根据需要查阅*Java*的*API*文档）。

**Context类使用的java.util.StringTokenizer的方法**

| 名字          | 说明                               |
| ------------- | ---------------------------------- |
| NextToken     | 获取下一个标记（前进至下一个标记） |
| hasMoreTokens | 检查是否还有下一个标记             |

```java
import java.util.StringTokenizer;

public class Context {
    private StringTokenizer tokenizer;
    private String currentToken;
    public Context(String text) {
        tokenizer = new StringTokenizer(text);
        nextToken();
    }
    public String nextToken() {
        if (tokenizer.hasMoreTokens()) {
            currentToken = tokenizer.nextToken();
        } else {
            currentToken = null;
        }
        return currentToken;
    }
    public String currentToken() {
        return currentToken;
    }
    public void skipToken(String token) throws ParseException {
        if (!token.equals(currentToken)) {
            throw new ParseException("Warning: " + token + " is expected, but" + currentToken + " is found.");
        }
        nextToken();
    }
    public int currentNumber() throws ParseException {
        int number = 0;
        try {
            number = Integer.parseInt(currentToken);
        } catch (NumberFormatException e) {
            throw new ParseException("Warning: " + e);
        }
        return number;
    }
}
```



> ParseException类

*ParseException*类是表示语法解析时可能发生的异常的类。该类比较简单，没有什么需要特别注意的地方。

```java
public class ParseException extends Exception {
    public ParseException(String msg) {
        super(msg);
    }
}
```



> Main类

*Main*类是启动我们之前学习的迷你语言解释器的程序。它会读取*program.txt*文件，然后逐行解析迷你程序，并将解析结果显示出来。

在显示结果中，以*“text =”*开头的部分是迷你程序语句，以*“node =”*开头的部分是语法解析结果。通过查看运行结果我们可以发现，语法解析器识别出了*program ... end*字符串中的迷你语言的语法元素，并为它们加上了[]。这表示语法解析器正确地理解了我们定义的迷你语言。

注意：将*CommandListNode*的实例转换为字符串显示出来——例如在*[go, right]*中加上大括号和逗号——的是*java.util.ArrayList*的*toString*方法。

```java
import java.io.BufferedReader;
import java.io.FileReader;

public class Main {
    public static void main(String[] args) {
        try {
            BufferedReader reader = new BufferedReader(new FileReader("program.txt"));
            String text;
            while ((text = reader.readLine()) != null) {
                System.out.println("text = \"" + text + "\"");
                Node node = new ProgramNode();
                node.parse(new Context(text));
                System.out.println("node = " + node);
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

```



### 23.4 Interpreter模式中的登场角色

+ ***AbstractExpression*（抽象表达式）**

*AbstractExpression*角色定义了语法树节点的共同接口（*API*）。在示例程序中，由*Node*类扮演此角色。在示例程序中，共同接口（*API*）的名字是*parse*，不过在图中它的名字是*interpreter*。

+ ***TermminalExpression*（终结符表达式）**

*TerminalExpression*角色对应*BNF*中的终结符表达式。在示例程序中，由*PrimitiveCommandNode*类扮演此角色。

+ ***NonterminalExpression*（非终结符表达式）**

*NonterminalExpression*角色对应*BNF*中的非终结符表达式。在示例程序中，由*ProgramNode*类、*CommandNode*类、*RepeatCommandNode*类和*CommandListNode*类扮演此角色。

+ ***Context*（文脉、上下文）**

*Context*角色为解释器进行语法解析提供了必要的信息。在示例程序中，由*Context*类扮演此角色。

+ ***Client*（请求者）**

为了推导语法树，*Client*角色会调用*TerminalExpression*角色和*NonterminalExpression*角色。在示例程序中，由*Main*类扮演此角色。



###  23.5 拓展思路的要点

> 还有其他哪些迷你语言

+ **正则表达式**

在*GoF*书中，作者使用正则表达式（*regular expression*）作为迷你语言示例。在书中，作者使用*Interpreter*模式解释了如下表达式，并推导出语法树。

`raining & (dogs | cats) *`

这个表达式的意思是“在*raining*后重复出现0次以上*dogs*或*cats*”。

+ **检索表达式**

在*Grand*书中，作者讲解了表示单词组合的*Little Language*模式。在书中，该模式可以解释如下表达式并推导出语法树。

`garlic and not onions`

这个表达式的意思是“包含*garlic*但不包含*onions*”。

+ **批处理语言**

*Interpreter*模式还可以处理批处理语言，即将基本命令组合在一起，并按顺序执行或是循环执行的语言。本章中的无线玩具车操控就是一种批处理语言。

> 跳过标记还是读取标记

在制作解释器时，经常会出现多读了一个标记或是漏读了一个标记的*Bug*。在编写各个终结符表达式对应的方法时，我们必须时刻注意“进入这个方法时已经读至哪个标记了？出了这个方法时应该读至哪个标记？”



### 23.6 相关的设计模式

+ ***Composite*模式**

*NonterminalExpression*角色多是递归结构，因此经常会使用*Composite*模式来实现*NonterminalExpression*角色。

+ ***Flyweight*模式**

有时会使用*Flyweight*模式来共享*TerminalExpression*角色。

+ ***Visitor*模式**

在推导出语法树后，有时会使用*Visitor*模式来访问语法树的各个节点。