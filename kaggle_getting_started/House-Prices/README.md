# House-Prices
<p>&nbsp;</p>
<p>　　今天看了个新闻，说是中国社会科学院城市发展与环境研究所及社会科学文献出版社共同发布《房地产蓝皮书：中国房地产发展报告No.16(2019)》指出房价上涨7.6%，看得我都坐不住了，这房价上涨什么时候是个头啊。为了让自己以后租得起房，我还是好好努力吧。于是我打开了Kaggle，准备上手第一道题，正巧发现有个房价预测，可能这是命运的安排吧......</p>
<h2>一、下载数据</h2>
<p>　　进入到&nbsp;kaggle&nbsp;后要先登录，需要注意的是，注册的时候有一个验证，要翻墙才会显示验证信息。</p>
<p>　　&nbsp;&nbsp;　　<img src="https://img2018.cnblogs.com/blog/1458123/201905/1458123-20190525190519940-1131080000.png" alt="" /></p>
<p>　　<img src="https://img2018.cnblogs.com/blog/1458123/201905/1458123-20190525190705565-1918222721.png" alt="" /></p>
<p>　　下载好数据之后，大致看一下数据的情况，在对应题目的页面也有关于数据属性的一些解释，看一下对应数据代表什么。</p>
<h2>二、数据预处理</h2>
<h3 id="提取-y_train-并做相应处理">　提取 y_train 并做相应处理</h3>
<p>　　先导入需要用到的包，通过 pandas&nbsp;的&nbsp;read_csv(filename, index_col=0)&nbsp;分别将测试集和训练集导入。完了之后，我们把训练集里的&ldquo;SalePrice&rdquo;取出来，查看它的分布情况并作一下处理。</p>
<div class="cnblogs_code">
<pre>y_train = train_data.pop(<span style="color: #800000;">'</span><span style="color: #800000;">SalePrice</span><span style="color: #800000;">'</span><span style="color: #000000;">)
y_train.hist()</span></pre>
</div>
<p>&nbsp; 　　　　&nbsp; &nbsp; &nbsp;&nbsp;<img src="https://img2018.cnblogs.com/blog/1458123/201905/1458123-20190525191803550-1295793262.png" alt="" /></p>
<p>　　由此可见数据并不平滑，因此需要将其正态化，正态化可以使数据变得平滑，目的是稳定方差，直线化，使数据分布正态或者接近正态。正态化可以用&nbsp;numpy&nbsp;的&nbsp;log1p()&nbsp;处理。log1p(y_train)&nbsp;可以理解为&nbsp;log 1&nbsp;plus，即 log（y_train + 1）。正态化之前，先看一下如果正态化之后的价格分布。</p>
<p>　　　　　　<img src="https://img2018.cnblogs.com/blog/1458123/201905/1458123-20190525192042234-761656383.png" alt="" /></p>
<p>　　这样的分布就很好了，因此我们通过 numpy 的 log1p() 将 y_train 正态化。</p>
<div class="cnblogs_code">
<pre>y_train = np.log1p(y_train)</pre>
</div>
<h3 id="将去掉-SalePrice-的训练集合测试集合并">　将去掉 SalePrice 的训练集和测试集合并</h3>
<p>　　为了将两个数据集一起处理，减少重复的步骤，将两个数据集合并再处理。使用 pandas 的 concat()&nbsp;将训练集和测试集合并起来并看一下合并后的数据的行数和列数，以确保正确合并。</p>
<div class="cnblogs_code">
<pre>data = pd.concat((train_data, test_data), axis=<span style="color: #000000;">0)
data.shape</span></pre>
</div>
<h3 id="特征处理">　特征处理</h3>
<p>　　数据集中有几个跟年份有关的属性，分别是：</p>
<ul>
<li>YrSold: 售出房子的年份；</li>
<li>YearBuilt：房子建成的年份；</li>
<li>YearRemodAdd：装修的年份；</li>
<li>GarageYrBlt：车库建成的年份</li>
</ul>
<p>　　算出跟售出房子的时间差，并新生成单独的列，然后删除这些年份</p>
<div class="cnblogs_code">
<pre>data.eval(<span style="color: #800000;">'</span><span style="color: #800000;">Built2Sold = YrSold-YearBuilt</span><span style="color: #800000;">'</span>, inplace=<span style="color: #000000;">True)
data.eval(</span><span style="color: #800000;">'</span><span style="color: #800000;">Add2Sold = YrSold-YearRemodAdd</span><span style="color: #800000;">'</span>, inplace=<span style="color: #000000;">True)
data.eval(</span><span style="color: #800000;">'</span><span style="color: #800000;">GarageBlt = YrSold-GarageYrBlt</span><span style="color: #800000;">'</span>, inplace=<span style="color: #000000;">True)
data.drop([</span><span style="color: #800000;">'</span><span style="color: #800000;">YrSold</span><span style="color: #800000;">'</span>, <span style="color: #800000;">'</span><span style="color: #800000;">YearBuilt</span><span style="color: #800000;">'</span>, <span style="color: #800000;">'</span><span style="color: #800000;">YearRemodAdd</span><span style="color: #800000;">'</span>, <span style="color: #800000;">'</span><span style="color: #800000;">GarageYrBlt</span><span style="color: #800000;">'</span>], axis=1, inplace=True)</pre>
</div>
<p>　　接下来进行变量转换，由于有一些列是类别型的，但由于pandas的特性，数字符号会被默认成数字。比如下面三列，是以数字来表示等级的，但被认为是数字，这样就会使得预测受到影响。</p>
<ul>
<li>OverallQual: Rates the overall material and finish of the house</li>
<li>OverallCond: Rates the overall condition of the house</li>
<li>MSSubClass: The building class</li>
</ul>
<p>　　这三个相当于是等级和类别，只不过是用数字来当等级的高低而已。因此我们要把这些转换成 string</p>
<div class="cnblogs_code">
<pre>data[<span style="color: #800000;">'</span><span style="color: #800000;">OverallQual</span><span style="color: #800000;">'</span>] = data[<span style="color: #800000;">'</span><span style="color: #800000;">OverallQual</span><span style="color: #800000;">'</span><span style="color: #000000;">].astype(str)
data[</span><span style="color: #800000;">'</span><span style="color: #800000;">OverallCond</span><span style="color: #800000;">'</span>] = data[<span style="color: #800000;">'</span><span style="color: #800000;">OverallCond</span><span style="color: #800000;">'</span><span style="color: #000000;">].astype(str)
data[</span><span style="color: #800000;">'</span><span style="color: #800000;">MSSubClass</span><span style="color: #800000;">'</span>] = data[<span style="color: #800000;">'</span><span style="color: #800000;">MSSubClass</span><span style="color: #800000;">'</span>].astype(str)</pre>
</div>
<h3 id="把category的变量转变成numerical">　把category的变量转变成numerical　</h3>
<p>　　我们可以用One-Hot的方法来表达category。pandas自带的get_dummies方法，可以一键做到One-Hot。</p>
<p>　　这里按我的理解解释一下One-Hot：比如说有一组自拟的数据 data，其中 data['学历要求']有'大专', '本科', '硕士', '不限'。但data['学历要求']=='本科'，则他可以用字典表示成这样{'大专': 0, '本科':1, '硕士':0, '不限':0}，用向量表示为[0, 1, 0, 0]</p>
<div class="cnblogs_code">
<pre>dummied_data = pd.get_dummies(data)</pre>
</div>
<p>　　　　　　<img src="https://img2018.cnblogs.com/blog/1458123/201905/1458123-20190525193917111-1236183083.png" alt="" /></p>
<h3>　处理numerical变量</h3>
<p>　　category变量处理好了之后，就该轮到numerical变量了。查看一下缺失值情况。</p>
<div class="cnblogs_code">
<pre>dummied_data.isnull().sum().sort_values(ascending=False).head()</pre>
</div>
<p>　　　　　　　　　　　<img src="https://img2018.cnblogs.com/blog/1458123/201905/1458123-20190525194145590-1928104646.png" alt="" /></p>
<p>　　上面的数据显示的是每列对应的缺失值情况，对于缺失值，需要进行填充，可以使用平均值进行填充。</p>
<div class="cnblogs_code">
<pre>mean_cols =<span style="color: #000000;"> dummied_data.mean()
dummied_data </span>= dummied_data.fillna(mean_cols)</pre>
</div>
<h3>　标准差标准化</h3>
<p>　　缺失值处理完毕，由于有一些数据的值比较大，特别是比起 one-hot 后的数值 0 和 1，那些几千的值就相对比较大了。因此对数值型变量进行标准化。</p>
<div class="cnblogs_code">
<pre>numerical_cols = data.columns[data.dtypes != <span style="color: #800000;">'</span><span style="color: #800000;">object</span><span style="color: #800000;">'</span>]  <span style="color: #008000;">#</span><span style="color: #008000;"> 数据为数值型的列名</span>
num_cols_mean =<span style="color: #000000;"> dummied_data.loc[:, numerical_cols].mean()
num_cols_std </span>=<span style="color: #000000;"> dummied_data.loc[:, numerical_cols].std()
dummied_data.loc[:, numerical_cols] </span>= (dummied_data.loc[:, numerical_cols] - num_cols_mean) / num_cols_std</pre>
</div>
<p>　　到这里，数据处理算是完毕了。虽然这样处理还不够完善，后面如果技术再精进一点可能会重新弄一下。接下来需要将数据集分开，分成训练集合测试集。</p>
<div class="cnblogs_code">
<pre>X_train =<span style="color: #000000;"> dummied_data.loc[train_data.index].values
X_test </span>= dummied_data.loc[test_data.index].values</pre>
</div>
<h2 id="建模预测">三、建模预测</h2>
<p>　　由于这是一个回归问题，我用 sklearn.selection&nbsp;的&nbsp;cross_val_score 试了岭回归（Ridge&nbsp;Regression）、BaggingRegressor 以及 XGBoost。且不说集成算法比单个回归模型好，XGBoost&nbsp; 不愧是&nbsp;Kaggle&nbsp;神器，效果比 BaggingRegressor&nbsp;还要好很多。安装&nbsp;XGBoost&nbsp; 的过程就不说了，安装好之后导入包就行了，但是我们还要调一下参。</p>
<div class="cnblogs_code">
<pre>params = [6,7,8<span style="color: #000000;">]
scores </span>=<span style="color: #000000;"> []
</span><span style="color: #0000ff;">for</span> param <span style="color: #0000ff;">in</span><span style="color: #000000;"> params:
    model </span>= XGBRegressor(max_depth=<span style="color: #000000;">param)
    score </span>= np.sqrt(-cross_val_score(model, X_train, y_train, cv=10, scoring=<span style="color: #800000;">'</span><span style="color: #800000;">neg_mean_squared_error</span><span style="color: #800000;">'</span><span style="color: #000000;">))
    scores.append(np.mean(score))
plt.plot(params, scores)</span></pre>
</div>
<p>　　　　　　　　　　<img src="https://img2018.cnblogs.com/blog/1458123/201905/1458123-20190525195538934-670126322.png" alt="" /></p>
<p>　　可见当&nbsp;max_depth = 7&nbsp;的时候，错误率最低。接下来就是建模训练预测了。</p>
<div class="cnblogs_code">
<pre>xgbr = XGBRegressor(max_depth=7<span style="color: #000000;">)
xgbr.fit(X_train, y_train)
y_prediction </span>= np.expm1(xgbr.predict(X_test))</pre>
</div>
<p>　　得到结果之后，将结果保存为 .csv&nbsp;文件，因为&nbsp;kaggle&nbsp;的提交要求是&nbsp;csv&nbsp;文件，可以到 kaggle&nbsp;看一下提交要求，前面下载的文件里面也有一份提交样式，照它的格式保存文件就好了。</p>
<div class="cnblogs_code">
<pre>submitted_data = pd.DataFrame(data= {<span style="color: #800000;">'</span><span style="color: #800000;">Id</span><span style="color: #800000;">'</span> : test_data.index, <span style="color: #800000;">'</span><span style="color: #800000;">SalePrice</span><span style="color: #800000;">'</span><span style="color: #000000;">: y_prediction})
submitted_data.to_csv(</span><span style="color: #800000;">'</span><span style="color: #800000;">./input/submission.csv</span><span style="color: #800000;">'</span>, index=False)　　<span style="color: #008000;">#</span><span style="color: #008000;"> 将预测结果保存到文件</span></pre>
</div>
<h2>&nbsp;四、提交结果</h2>
<p>　　　　　　<img src="https://img2018.cnblogs.com/blog/1458123/201905/1458123-20190525200425480-1835568673.png" alt="" /></p>
<p>　　　　　　&nbsp;<img src="https://img2018.cnblogs.com/blog/1458123/201905/1458123-20190525200749846-1177775138.png" alt="" /></p>
<p>　　提交的时候也要翻墙，才会上传文件。到这里就结束了。</p>
<p>&nbsp;</p>
<p>　　想要第一时间获取更多有意思的推文，可关注公众号：&nbsp;Max的日常操作</p>
<p>　　　　　　　　　　　　　　　　　　　　　　　　　　　&nbsp;<img src="https://img2018.cnblogs.com/blog/1458123/201905/1458123-20190522173941723-1301336312.png" alt="" /></p>
<p>&nbsp;</p>
