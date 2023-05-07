# log linear model

## t1
online training
## t3
sgd training

# result 30 epoch

```commandline
DataLoader(
  num of sentences: 803
  num of uniq words: 4537
  num of uniq tags: 31
  num of uniq chars: 1896
)
DataLoader(
  num of sentences: 1910
  num of uniq words: 8577
  num of uniq tags: 30
  num of uniq chars: 2524
)

---feature space constructing over---
---training set---
num correctly-predict sent: 0.0547945205479452, acc over all tag: 0.660848733744011
---dev set---
num correctly-predict sent: 0.04659685863874346, acc over all tag: 0.6222301715057931
---training set---
num correctly-predict sent: 0.07098381070983811, acc over all tag: 0.7468465825755354
---dev set---
num correctly-predict sent: 0.05602094240837696, acc over all tag: 0.698702279457064
---training set---
num correctly-predict sent: 0.08094645080946451, acc over all tag: 0.7987679671457906
---dev set---
num correctly-predict sent: 0.061780104712041886, acc over all tag: 0.7399193147717562
---training set---
num correctly-predict sent: 0.09464508094645081, acc over all tag: 0.8302532511978097
---dev set---
num correctly-predict sent: 0.06701570680628273, acc over all tag: 0.7652179097358851
---training set---
num correctly-predict sent: 0.12453300124533001, acc over all tag: 0.8537694338515693
---dev set---
num correctly-predict sent: 0.0774869109947644, acc over all tag: 0.7821101373238737
---training set---
num correctly-predict sent: 0.14196762141967623, acc over all tag: 0.8683876014471497
---dev set---
num correctly-predict sent: 0.08324607329842931, acc over all tag: 0.7911921938035335
---training set---
num correctly-predict sent: 0.1718555417185554, acc over all tag: 0.8805123692187348
---dev set---
num correctly-predict sent: 0.09005235602094241, acc over all tag: 0.798863252449373
---training set---
num correctly-predict sent: 0.19053549190535493, acc over all tag: 0.8915615527525178
---dev set---
num correctly-predict sent: 0.093717277486911, acc over all tag: 0.8072696198255133
---training set---
num correctly-predict sent: 0.20921544209215442, acc over all tag: 0.9008506893517161
---dev set---
num correctly-predict sent: 0.0968586387434555, acc over all tag: 0.8128142451161589
---training set---
num correctly-predict sent: 0.23536737235367372, acc over all tag: 0.9093575828688765
---dev set---
num correctly-predict sent: 0.09947643979057591, acc over all tag: 0.8178620401836284
---training set---
num correctly-predict sent: 0.2465753424657534, acc over all tag: 0.9151755157915322
---dev set---
num correctly-predict sent: 0.10209424083769633, acc over all tag: 0.8220950336850892
---training set---
num correctly-predict sent: 0.26525529265255293, acc over all tag: 0.9207978879436785
---dev set---
num correctly-predict sent: 0.10732984293193717, acc over all tag: 0.8255728452473221
---training set---
num correctly-predict sent: 0.28393524283935245, acc over all tag: 0.9254913464359049
---dev set---
num correctly-predict sent: 0.11151832460732984, acc over all tag: 0.8298058387487828
---training set---
num correctly-predict sent: 0.3001245330012453, acc over all tag: 0.9297936833871125
---dev set---
num correctly-predict sent: 0.11413612565445026, acc over all tag: 0.8328861861324748
---training set---
num correctly-predict sent: 0.31133250311332505, acc over all tag: 0.9334115576415372
---dev set---
num correctly-predict sent: 0.11675392670157068, acc over all tag: 0.8348536338162523
---training set---
num correctly-predict sent: 0.32254047322540474, acc over all tag: 0.9365894201623154
---dev set---
num correctly-predict sent: 0.1193717277486911, acc over all tag: 0.8374371509767682
---training set---
num correctly-predict sent: 0.33748443337484435, acc over all tag: 0.9393272709494476
---dev set---
num correctly-predict sent: 0.12198952879581151, acc over all tag: 0.8394642182873269
---training set---
num correctly-predict sent: 0.3449564134495641, acc over all tag: 0.9419184511586975
---dev set---
num correctly-predict sent: 0.12356020942408377, acc over all tag: 0.840398259106898
---training set---
num correctly-predict sent: 0.3574097135740971, acc over all tag: 0.9442651804048108
---dev set---
num correctly-predict sent: 0.12408376963350785, acc over all tag: 0.8415111588068125
---training set---
num correctly-predict sent: 0.3698630136986301, acc over all tag: 0.9469541409993155
---dev set---
num correctly-predict sent: 0.1256544502617801, acc over all tag: 0.8429420298495598
---training set---
num correctly-predict sent: 0.37359900373599003, acc over all tag: 0.9488119683191552
---dev set---
num correctly-predict sent: 0.12513089005235603, acc over all tag: 0.8445716329815776
---training set---
num correctly-predict sent: 0.3835616438356164, acc over all tag: 0.9506697956389948
---dev set---
num correctly-predict sent: 0.12513089005235603, acc over all tag: 0.8456845326814921
---training set---
num correctly-predict sent: 0.386052303860523, acc over all tag: 0.9518431602620514
---dev set---
num correctly-predict sent: 0.12460732984293194, acc over all tag: 0.8463800949939386
---training set---
num correctly-predict sent: 0.3972602739726027, acc over all tag: 0.9531143052703628
---dev set---
num correctly-predict sent: 0.12513089005235603, acc over all tag: 0.8474333750670721
---training set---
num correctly-predict sent: 0.40473225404732255, acc over all tag: 0.9548254620123203
---dev set---
num correctly-predict sent: 0.12774869109947645, acc over all tag: 0.8489039925276735
---training set---
num correctly-predict sent: 0.42092154420921546, acc over all tag: 0.9564877285616505
---dev set---
num correctly-predict sent: 0.12774869109947645, acc over all tag: 0.849997019018661
---training set---
num correctly-predict sent: 0.42714819427148193, acc over all tag: 0.9573188618363156
---dev set---
num correctly-predict sent: 0.12827225130890052, acc over all tag: 0.8506925813311076
---training set---
num correctly-predict sent: 0.43462017434620176, acc over all tag: 0.9588344578077638
---dev set---
num correctly-predict sent: 0.12879581151832462, acc over all tag: 0.8511695383453566
---training set---
num correctly-predict sent: 0.44333748443337484, acc over all tag: 0.9600567126234477
---dev set---
num correctly-predict sent: 0.12879581151832462, acc over all tag: 0.8515868757328245
---training set---
num correctly-predict sent: 0.4520547945205479, acc over all tag: 0.961181187053877
---dev set---
num correctly-predict sent: 0.12931937172774868, acc over all tag: 0.8525407897613227
---online training over---

```