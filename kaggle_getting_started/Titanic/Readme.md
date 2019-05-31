# Titanic

<p>　　首先导入相应的模块，然后检视一下数据情况。对数据有一个大致的了解之后，开始进行下一步操作。</p>
<p>　　<img src="https://img2018.cnblogs.com/blog/1458123/201905/1458123-20190531174727887-829650175.png" alt="" /></p>
<h2>一、分析数据</h2>
<h3>　1、Survived 的情况</h3>
<div class="cnblogs_code">
<pre>train_data[<span style="color: #800000;">'</span><span style="color: #800000;">Survived</span><span style="color: #800000;">'</span>].value_counts()　</pre>
</div>
<h3>　　<img src="https://img2018.cnblogs.com/blog/1458123/201905/1458123-20190531175118596-1814711908.png" alt="" width="341" height="80" /></h3>
<h3>　2、Pclass 和 Survived 之间的关系</h3>
<div class="cnblogs_code">
<pre>train_data.groupby(<span style="color: #800000;">'</span><span style="color: #800000;">Pclass</span><span style="color: #800000;">'</span>)[<span style="color: #800000;">'</span><span style="color: #800000;">Survived</span><span style="color: #800000;">'</span>].mean()</pre>
</div>
<p>&nbsp;　　<img src="https://img2018.cnblogs.com/blog/1458123/201905/1458123-20190531175225820-1782351616.png" alt="" /></p>
<h3>　3、Embarked 和 Survived 之间的关系</h3>
<div class="cnblogs_code">
<pre>train_data.groupby(<span style="color: #800000;">'</span><span style="color: #800000;">Embarked</span><span style="color: #800000;">'</span>)[<span style="color: #800000;">'</span><span style="color: #800000;">Survived</span><span style="color: #800000;">'</span><span style="color: #000000;">].value_counts()
sns.countplot(</span><span style="color: #800000;">'</span><span style="color: #800000;">Embarked</span><span style="color: #800000;">'</span>,hue=<span style="color: #800000;">'</span><span style="color: #800000;">Survived</span><span style="color: #800000;">'</span>,data=train_data)</pre>
</div>
<p>　　<img src="https://img2018.cnblogs.com/blog/1458123/201905/1458123-20190531175417360-1544406220.png" alt="" /></p>
<h2>二、特征处理</h2>
<p>　　先将 label 提取出来，然后将 train 和 test 合并起来一起处理。</p>
<div class="cnblogs_code">
<pre>y_train = train_data.pop(<span style="color: #800000;">'</span><span style="color: #800000;">Survived</span><span style="color: #800000;">'</span><span style="color: #000000;">).astype(str).values
data </span>= pd.concat((train_data, test_data), axis=0)</pre>
</div>
<h3>　1、对 numerical 数据进行处理</h3>
<p>　　（1）SibSp/Parch （兄弟姐妹配偶数 / 父母孩子数）</p>
<p>　　由于这两个属性都和 Survived 没有很大的影响，将这两个属性的值相加，表示为家属个数。</p>
<div class="cnblogs_code">
<pre>data[<span style="color: #800000;">'</span><span style="color: #800000;">FamilyNum</span><span style="color: #800000;">'</span>] = data[<span style="color: #800000;">'</span><span style="color: #800000;">SibSp</span><span style="color: #800000;">'</span>] + data[<span style="color: #800000;">'</span><span style="color: #800000;">Parch</span><span style="color: #800000;">'</span>]</pre>
</div>
<p>　　（2）Fare （费用）</p>
<p>　　它有一个缺失值，需要将其补充。(这里是参考别人的，大神总能发现一些潜在的信息：票价和 Pclass 和 Embarked 有关)&nbsp; 因此，先看一下他们之间的关系以及缺失值的情况。</p>
<div class="cnblogs_code">
<pre>train_data.groupby(by=[<span style="color: #800000;">"</span><span style="color: #800000;">Pclass</span><span style="color: #800000;">"</span>,<span style="color: #800000;">"</span><span style="color: #800000;">Embarked</span><span style="color: #800000;">"</span>]).Fare.mean()</pre>
</div>
<p>　　　　<img src="https://img2018.cnblogs.com/blog/1458123/201905/1458123-20190531180114164-2126723737.png" alt="" /></p>
<p>　　缺失值 Pclass = 3， Embarked = S，因此我们将其置为14.644083.</p>
<div class="cnblogs_code">
<pre>data[<span style="color: #800000;">"</span><span style="color: #800000;">Fare</span><span style="color: #800000;">"</span>].fillna(14.644083,inplace=True)</pre>
</div>
<p>　　还有 Age 的缺失值也需要处理，我是直接将其设置为平均值。</p>
<h3>　2、对 categorical 数据进行处理</h3>
<p>　　（1）对 Cabin 进行处理</p>
<p>　　Cabin虽然有很多空值，但他的值的开头都是字母，按我自己的理解应该是对应船舱的位置，所以取首字母。考虑到船舱位置对救生是有一定影响的，虽然有很多缺失值，但还是把它保留下来，而且由于 T 开头的只有一条数据，因此将它设置成数量较小的 G。</p>
<div class="cnblogs_code">
<pre>data[<span style="color: #800000;">'</span><span style="color: #800000;">Cabin</span><span style="color: #800000;">'</span>] = data[<span style="color: #800000;">'</span><span style="color: #800000;">Cabin</span><span style="color: #800000;">'</span><span style="color: #000000;">].str[0]
data[</span><span style="color: #800000;">'</span><span style="color: #800000;">Cabin</span><span style="color: #800000;">'</span>][data[<span style="color: #800000;">'</span><span style="color: #800000;">Cabin</span><span style="color: #800000;">'</span>]==<span style="color: #800000;">'</span><span style="color: #800000;">T</span><span style="color: #800000;">'</span>] = <span style="color: #800000;">'</span><span style="color: #800000;">G</span><span style="color: #800000;">'</span></pre>
</div>
<p>　　（2）对 Ticket 进行处理</p>
<p>　　将 Ticket 的头部取出来当成新列。</p>
<div class="cnblogs_code">
<pre>data[<span style="color: #800000;">'</span><span style="color: #800000;">Ticket_Letter</span><span style="color: #800000;">'</span>] = data[<span style="color: #800000;">'</span><span style="color: #800000;">Ticket</span><span style="color: #800000;">'</span><span style="color: #000000;">].str.split().str[0]
data[</span><span style="color: #800000;">'</span><span style="color: #800000;">Ticket_Letter</span><span style="color: #800000;">'</span>] = data[<span style="color: #800000;">'</span><span style="color: #800000;">Ticket_Letter</span><span style="color: #800000;">'</span>].apply(<span style="color: #0000ff;">lambda</span> x:np.nan <span style="color: #0000ff;">if</span> x.isnumeric() <span style="color: #0000ff;">else</span><span style="color: #000000;"> x)
data.drop(</span><span style="color: #800000;">'</span><span style="color: #800000;">Ticket</span><span style="color: #800000;">'</span>,inplace=True,axis=1)</pre>
</div>
<p>　　（3）对 Name 进行处理</p>
<p>　　名字这个东西，虽然它里面的称呼可能包含了一些身份信息，但我还是打算把这一列给删掉...</p>
<div class="cell border-box-sizing code_cell rendered">
<div class="cnblogs_code">
<pre>data.drop(<span style="color: #800000;">'</span><span style="color: #800000;">Name</span><span style="color: #800000;">'</span>,inplace=True,axis=1)</pre>
</div>
<p>　　（4）统一将 categorical 数据进行 One-Hot</p>
<p>　　One-Hot 大致的意思在之前的文章讲过了，这里也不再赘述。</p>
<div class="cnblogs_code">
<pre>data[<span style="color: #800000;">'</span><span style="color: #800000;">Pclass</span><span style="color: #800000;">'</span>] = data[<span style="color: #800000;">'</span><span style="color: #800000;">Pclass</span><span style="color: #800000;">'</span><span style="color: #000000;">].astype(str)
data[</span><span style="color: #800000;">'</span><span style="color: #800000;">FamilyNum</span><span style="color: #800000;">'</span>] = data[<span style="color: #800000;">'</span><span style="color: #800000;">FamilyNum</span><span style="color: #800000;">'</span><span style="color: #000000;">].astype(str)
dummied_data </span>= pd.get_dummies(data)</pre>
</div>
<p>　　（5）数据处理完毕，将训练集和测试集分开</p>
<div class="cnblogs_code">
<pre>X_train =<span style="color: #000000;"> dummied_data.loc[train_data.index].values
X_test </span>= dummied_data.loc[test_data.index].values</pre>
</div>
<h2>三、构建模型</h2>
<p>　　这里用到了&nbsp;<span class="nn">sklearn.model_selection <span class="k">的&nbsp;<span class="n">GridSearchCV，我主要用它来调参以及评定 score。</span></span></span></p>
<h3>　1、XGBoost</h3>
<div class="cnblogs_code">
<pre>xgbc =<span style="color: #000000;"> XGBClassifier()
params </span>= {<span style="color: #800000;">'</span><span style="color: #800000;">n_estimators</span><span style="color: #800000;">'</span>: [100,110,120,130,140<span style="color: #000000;">], 
          </span><span style="color: #800000;">'</span><span style="color: #800000;">max_depth</span><span style="color: #800000;">'</span>:[5,6,7,8,9<span style="color: #000000;">]}
clf </span>= GridSearchCV(xgbc, params, cv=5, n_jobs=-1<span style="color: #000000;">)
clf.fit(X_train, y_train)
</span><span style="color: #0000ff;">print</span><span style="color: #000000;">(clf.best_params_)
</span><span style="color: #0000ff;">print</span>(clf.best_score_)　</pre>
</div>
<pre>　　<span style="font-size: 14px;">{'max_depth': 6, 'n_estimators': 130}
　　0.835016835016835 <br /></span></pre>
</div>
<h3>　2、Random Forest</h3>
<div class="cnblogs_code">
<pre>rf =<span style="color: #000000;"> RandomForestClassifier()
params </span>=<span style="color: #000000;"> {
    </span><span style="color: #800000;">'</span><span style="color: #800000;">n_estimators</span><span style="color: #800000;">'</span>: [100,110,120,130,140,150<span style="color: #000000;">],
    </span><span style="color: #800000;">'</span><span style="color: #800000;">max_depth</span><span style="color: #800000;">'</span>: [5,6,7,8,9,10<span style="color: #000000;">],
}
clf </span>= GridSearchCV(rf, params, cv=5, n_jobs=-1<span style="color: #000000;">)
clf.fit(X_train, y_train)
</span><span style="color: #0000ff;">print</span><span style="color: #000000;">(clf.best_params_)
</span><span style="color: #0000ff;">print</span>(clf.best_score_)</pre>
</div>
<pre><span style="font-size: 14px;">　　{'max_depth': 8, 'n_estimators': 110}
　　0.8294051627384961<br /></span></pre>
<h2>四、模型融合</h2>
<div class="cnblogs_code">
<pre><span style="color: #0000ff;">from</span> sklearn.ensemble <span style="color: #0000ff;">import</span><span style="color: #000000;"> VotingClassifier
xgbc </span>= XGBClassifier(n_estimators=130, max_depth=6<span style="color: #000000;">)
rf </span>= RandomForestClassifier(n_estimators=110, max_depth=8<span style="color: #000000;">)

vc </span>= VotingClassifier(estimators=[(<span style="color: #800000;">'</span><span style="color: #800000;">rf</span><span style="color: #800000;">'</span>, rf),(<span style="color: #800000;">'</span><span style="color: #800000;">xgb</span><span style="color: #800000;">'</span>,xgbc)], voting=<span style="color: #800000;">'</span><span style="color: #800000;">hard</span><span style="color: #800000;">'</span><span style="color: #000000;">)
vc.fit(X_train, y_train)</span></pre>
</div>
<p>　　准备就绪，预测并保存模型与结果</p>
<div class="cnblogs_code">
<pre>y_test =<span style="color: #000000;"> vc.predict(X_test)

</span><span style="color: #008000;">#</span><span style="color: #008000;"> 保存模型</span>
<span style="color: #0000ff;">from</span> sklearn.externals <span style="color: #0000ff;">import</span><span style="color: #000000;"> joblib
joblib.dump(vc, </span><span style="color: #800000;">'</span><span style="color: #800000;">vc.pkl</span><span style="color: #800000;">'</span><span style="color: #000000;">)

submit </span>= pd.DataFrame(data= {<span style="color: #800000;">'</span><span style="color: #800000;">PassengerId</span><span style="color: #800000;">'</span> : test_data.index, <span style="color: #800000;">'</span><span style="color: #800000;">Survived</span><span style="color: #800000;">'</span><span style="color: #000000;">: y_test})
submit.to_csv(</span><span style="color: #800000;">'</span><span style="color: #800000;">./input/submit.csv</span><span style="color: #800000;">'</span>, index=False)</pre>
</div>
<p>　　最后提交即可。Over~&nbsp; &nbsp; </p>
<p>　　</p>
<p>　　想要第一时间获取更多有意思的推文，可关注公众号：&nbsp;Max的日常操作</p>
<p>　　　　　　　　　　　　　　　　　　　　　　　　　　　&nbsp;<img src="https://img2018.cnblogs.com/blog/1458123/201905/1458123-20190522173941723-1301336312.png" alt="" /></p>
<p>&nbsp;</p>
