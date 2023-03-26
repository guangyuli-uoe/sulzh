cd su_lzh  
git init  
cat ~/.ssh/id_rsa.pub  
git branch -M main  
git remote add origin git@github.com:...  
  
touch README.md  
git add README.md  
git commit -m ''  
git push -u origin main  
  
conda create -n suda python=3.8  

=============  
conll dataset  
=============  
每个词占一行，  
每行的第2列为当前词语，  
第4列为当前词的词性，  
第7列为当前词的中心词的序号，  
第8列为当前词语与中心词的依存关系。  
句子与句子之间以空行间隔

