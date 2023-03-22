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
