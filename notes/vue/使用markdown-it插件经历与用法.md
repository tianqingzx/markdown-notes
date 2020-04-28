@[toc]

## 关于为什么使用markdown-it插件
[这里是github官网，点击进入](https://github.com/markdown-it/markdown-it#readme)
首先在npm插件官网上看了一下，当然上面的链接也是在[npm官网](https://www.npmjs.com/package/markdown-itv)进入的。
![在npm官网中的下载量和相关信息](https://img-blog.csdnimg.cn/20200411154852880.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ1MTM4OTM2,size_16,color_FFFFFF,t_70)
可以看到mrakdown-it插件的下载量是相当高的，包括我一开始曾尝试过使用mavon-editor插件来实现自己的博客，但是很遗憾的是这个插件的使用本身出了很大问题，而且不太适合我的需求。
其实mavon-editor本身也是使用的markdown-it来实现的：
![mavon-editor的依赖项](https://img-blog.csdnimg.cn/20200411155522676.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ1MTM4OTM2,size_16,color_FFFFFF,t_70)
所以最终干脆选择了markdown-it来直接实现博客模块。

## 中途的问题
在使用markdown-it的过程中出现了相当多的问题，光是看官方文档是不足以真正使用该插件的。
[点击进入markdown-it中文官网](https://markdown-it.docschina.org/api/Core.html)
更多是通过debugger进入代码进行查看:
![debugger](https://img-blog.csdnimg.cn/20200411172948947.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ1MTM4OTM2,size_16,color_FFFFFF,t_70)
在调式的过程中突然发现了一个特殊的变量 state
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200411173535687.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ1MTM4OTM2,size_16,color_FFFFFF,t_70)
state变量中保存着原本读取出的markdown文本，同时也是在源代码中出现的重要的一个变量，该变量记录着多个数值。这个变量是一个*StateBlock*（`state_block.js`之中）实例。

之后在查阅源代码再结合官方文档的解释，再加上其他人的博客解释才发现本身markdown-it整个过程只要存在两个过程：
+ 解析过程：定义在目录`ruler_block`、`ruler_inline`、`ruler_core`中
+ 渲染过程：主要定义在根目录的js文件`renderer.js`中，但是该过程打开源码后比起解析过程来说几乎简单的过分：
```js
var default_rules = {};

default_rules.code_inline = function (tokens, idx, options, env, slf) {
  code ...
};

default_rules.code_block = function (tokens, idx, options, env, slf) {
  var token = tokens[idx];
  return  '<pre' + slf.renderAttrs(token) + '><code>' +
          escapeHtml(tokens[idx].content) +
          '</code></pre>\n';
};

default_rules.fence = function (tokens, idx, options, env, slf) {
	code ...
}

...
```
> 所以我判断，官方肯定会留下对于DOM的渲染接口和方式交给我们自己实现。

但是在百度了之后却没有得到什么好的，或者说足够好的方式，所以我只好重新开始了debugger和查阅源码的方法进行寻找。       

在这里我注意到了官方文档中的一个有意思的参数：`tokens`参数，在用*console.log(tokens)*后，果然发现了一些有趣的东西：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200411175929681.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ1MTM4OTM2,size_16,color_FFFFFF,t_70)
在`tokens`中包含了所有的节点元素解析好之后的值，同时也有可以进行复写的规则名称，你可以这样复写：
```js
module.exports = function cssPlugin (md, options) {
	var blockquoteRenderer = function (tokens, idx, options, env, renderer) {
    	return '<blockquote style="color: blue; padding-left: 10px">'
  	}
  	md.renderer.rules.blockquote_open = blockquoteRenderer
}
```
在这里基本每一种DOM元素节点都会对应着两种规则`[DOM_NAME]_open`、`[DOM_NAME]_close`，你可以任意改写渲染方式。

当然如果仅仅只是给原本难看的界面加点样式之类的，还可以使用另外一种方式：
> 使用 Token.attrSet()，这里就可以查阅官方中文文档了解了。
```js
module.exports = function cssPlugin (md, options) {
	var blockquoteRenderer = function (tokens, idx, options, env, renderer) {
    	tokens[idx].attrSet('style', 'color: blue; padding-left: 10px')
    	return md.renderer.renderToken(tokens, idx, options, env, renderer)
	}
  	md.renderer.rules.blockquote_open = blockquoteRenderer
}
```
在这里基本之前的问题就得到了解决了，之后还出一篇讲Katex渲染的文章。