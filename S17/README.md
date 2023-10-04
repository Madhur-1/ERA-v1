# Session 17

## Introduction

This assignment focusses on improving the previous code in terms of training time. We implement:
1. One Cycle Policy
2. 16-bit mixed Precision
3. Removed data with more than 150 tokens
4. Dynamic Padding Collate function

### Target
1. Achieve Loss < 1.8

### Metrics
Final Train Loss: 1.552
Model trained for 40 epochs

## Data Exploration

The dataset is the Opus Books dataset, involving translation from English to French.

```
SOURCE: "Just leave me alone, will you?" he said, going to bed and turning his back.
TARGET: —Fiche-moi la paix, hein! lui dit-il en se couchant et en tournant le dos.

SOURCE: – Je suis bien jeune, monsieur, pour que l’on veuille m’écouter ; il faudrait un ordre écrit de votre main.
TARGET: "I am very young, sir, to make them listen to me; I ought to have a written order from you."

SOURCE: "Will not monsieur take any supper to-night?"
TARGET: «Monsieur soupera-t-il ce soir?»

SOURCE: "Hardly," was the answer. "I have called it insuperable, and I speak advisedly."
TARGET: -- Ce sera difficile; j'ai dit qu'il était insurmontable, et je ne parle pas au hasard.»
```

## Learning Curves

<img width="415" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/6a11e17b-95a2-4265-9aa4-f203ecb73a65">

<img width="418" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/68213db7-2f44-4c10-a491-11b92c52e0d1">


## Training Log

```
EPOCH: 0, Loss: 6.790328025817871
--------------------------------------------------------------------------------
SOURCE: The flood was drowning his mane, and his cry of distress never ceased; he uttered it more hoarsely, with his large open mouth stretched out.
TARGET: Son cri de détresse ne cessait pas, le flot noyait sa criniere, qu'il le poussait plus rauque, de sa bouche tendue et grande ouverte.
PREDICTED: La poitrine était sa femme , et sa voix ne pas , il se leva , et lui , avec ses yeux .
--------------------------------------------------------------------------------
SOURCE: For the first time, he was received by them with civility.
TARGET: Pour la première fois, il en fut reçu avec politesse.
PREDICTED: En effet , il y avait , il avait avec les .
--------------------------------------------------------------------------------
SOURCE: Let me look at you; let me contemplate you!"
TARGET: Laissez que je vous voie, que je vous contemple!
PREDICTED: - vous , je me !
--------------------------------------------------------------------------------
SOURCE: I hoisted and attached myself to the same place, dividing my wonderment between the storm and this incomparable man who faced it head-on.
TARGET: Je m'y étais hissé et attaché aussi, partageant mon admiration entre cette tempête et cet homme incomparable qui lui tenait tête.
PREDICTED: Je me rappelle et de me voir mon père , mon esprit et la tête qui la tête qui lui .
EPOCH: 1, CER: 0.6691739559173584, WER: 1.0598552227020264, BLEU: 0.0
EPOCH: 1, Loss: 4.870634078979492
EPOCH: 2, Loss: 4.1518874168396
--------------------------------------------------------------------------------
SOURCE: The explanation of this fact could only be produced from the engineer's own lips, and they must wait for that till speech returned.
TARGET: On ne pourrait avoir l'explication de ce fait que de la bouche de l'ingénieur. Il fallait pour cela attendre que la parole lui fût revenue.
PREDICTED: L ' explication de cette explication ne pouvait que l ' ingénieur ne pas attendre à l ' ingénieur , et ils devaient attendre jusqu ' à ce mot .
--------------------------------------------------------------------------------
SOURCE: In front of us stood the pilothouse, and unless I'm extremely mistaken, Captain Nemo must have been inside, steering his Nautilus himself.
TARGET: Devant nous s'élevait la cage du timonier, et je me trompe fort, ou le capitaine Nemo devait être là, dirigeant lui-même son _Nautilus_.
PREDICTED: En avant de nous , le fanal était en ce moment , et je ne m ' aurais pas eu eu l ' idée de nous faire , en se .
--------------------------------------------------------------------------------
SOURCE: "And besides, the worry, the expense!
TARGET: -- Et, d’ailleurs, les embarras, la dépense...
PREDICTED: -- Et d ' ailleurs , les cordes , les cordes !
--------------------------------------------------------------------------------
SOURCE: "Well?"
TARGET: -- Eh bien?
PREDICTED: -- Eh bien ?
EPOCH: 3, CER: 0.6201037764549255, WER: 1.1073648929595947, BLEU: 0.0
EPOCH: 3, Loss: 3.758406162261963
EPOCH: 4, Loss: 3.5401690006256104
--------------------------------------------------------------------------------
SOURCE: I owe everything to him!
TARGET: Je lui doistout!
PREDICTED: Je lui dois tout le croire !
--------------------------------------------------------------------------------
SOURCE: – Il ne commettra plus la faute de passer par-dessus les murs, dit Clélia ; mais il sortira par la porte, s’il est acquitté.
TARGET: "He will not make the mistake of going over the walls again," said Clelia, "but he will leave by the door if he is acquitted."
PREDICTED: " He could not be the of the Corso ," said to the beginning ; " he was off , he was off at the door , he was .
--------------------------------------------------------------------------------
SOURCE: He rose and came towards me, and I saw his face all kindled, and his full falcon-eye flashing, and tenderness and passion in every lineament.
TARGET: Il se leva et s'avança vers moi; sa figure était brûlante, ses yeux de faucon brillaient; chacun de ses traits annonçait la tendresse et la passion.
PREDICTED: Il se leva vers moi , et je vis toute sa face , et son œil , , et de passion .
--------------------------------------------------------------------------------
SOURCE: Everything had disappeared in the midst of the vapour, the hot coal grew pale, and the women were nothing more than shadows with broken gestures.
TARGET: Tout avait disparu au milieu de la vapeur, le charbon pâlissait, les femmes n'étaient plus que des ombres aux gestes cassés.
PREDICTED: Tout avait disparu dans le milieu de la bataille , le charbon pâle , et les femmes ne s ' plus que des ombres .
EPOCH: 5, CER: 0.5987424254417419, WER: 1.0403575897216797, BLEU: 0.0
EPOCH: 5, Loss: 3.421630382537842
EPOCH: 6, Loss: 3.357682466506958
--------------------------------------------------------------------------------
SOURCE: God forgive me, how my heart bounded for joy, when hers, which was within touch of it, was breaking with sorrow!
TARGET: Que Dieu me pardonne, mais mon coeur bondit de joie, tandis que le sien, qui était tout contre, se brisait de douleur.
PREDICTED: - moi , mon coeur battait à mes yeux , lorsque le coeur de Dieu , c ’ était moi !
--------------------------------------------------------------------------------
SOURCE: On her part Therese seemed to revel in daring.
TARGET: Elle n'avait pas une hésitation, pas une peur.
PREDICTED: Thérèse lui semblait que Thérèse fût dans son immobilité .
--------------------------------------------------------------------------------
SOURCE: "Brother," said Jehan timidly, "I am come to see you."
TARGET: « Mon frère, dit timidement Jehan, je viens vous voir. »
PREDICTED: « Mon frère , dit timidement timidement timidement , je veux vous voir .
--------------------------------------------------------------------------------
SOURCE: The admission of additional water was enough to shift its balance.
TARGET: Une introduction d'eau avait suffi pour rompre son équilibre.
PREDICTED: La plus d ' eau fut assez assez rapide pour mettre sa chemise .
EPOCH: 7, CER: 0.6384326219558716, WER: 1.0973180532455444, BLEU: 0.0
EPOCH: 7, Loss: 3.3343355655670166
EPOCH: 8, Loss: 3.2895548343658447
--------------------------------------------------------------------------------
SOURCE: CHAPTER 8 What Is the Decoration that Confers Distinction?
TARGET: Chapitre VIII. Quelle est la décoration qui distingue ?
PREDICTED: Qu ' est - ce que le 30 août ?
--------------------------------------------------------------------------------
SOURCE: He stood considering me some minutes; then added, "She looks sensible, but not at all handsome."
TARGET: Il me regarda quelques minutes, puis ajouta: «Sa figure exprime la sensibilité, mais elle n'est pas jolie.
PREDICTED: -- Il me restait à quelques minutes ; puis elle ajouta : « Mais elle n ' est pas tres belle .
--------------------------------------------------------------------------------
SOURCE: "Away!" he cried harshly; "keep at a distance, child; or go in to Sophie!"
TARGET: «Éloigne-toi d'ici, enfant, s'écria-t-il durement, ou va jouer avec Sophie!»
PREDICTED: « De loin ! s ' écria - t - il en soupirant : « , une enfant ou de loin ! »
--------------------------------------------------------------------------------
SOURCE: Iremember that because we had a good laugh over it afterward.
TARGET: Je me rappelle cela parce que nous en avons beaucoup ri, plustard.
PREDICTED: que nous avions un rire quelconque .
EPOCH: 9, CER: 0.645999014377594, WER: 1.1056619882583618, BLEU: 0.0
EPOCH: 9, Loss: 3.1965136528015137
EPOCH: 10, Loss: 3.105790376663208
--------------------------------------------------------------------------------
SOURCE: For two hours our fishing proceeded energetically but without bringing up any rarities.
TARGET: Pendant deux heures, notre pêche fut activement conduite, mais sans rapporter aucune rareté.
PREDICTED: Pendant deux heures , nous fîmes des progrès sans lever la netteté .
--------------------------------------------------------------------------------
SOURCE: Both understood that they must accept the position without hesitation, and finish the business at one stroke.
TARGET: Ils comprirent tous deux qu'il fallait accepter la position sans hésiter et en finir d'un coup.
PREDICTED: On comprit qu ' on devait naturellement le but sans hésiter , et , en finir .
--------------------------------------------------------------------------------
SOURCE: 'A likely story indeed!' said the Pigeon in a tone of the deepest contempt.
TARGET: « Voilà une histoire bien vraisemblable ! » dit le Pigeon d’un air de profond mépris.
PREDICTED: « C ’ est une affaire , » dit le Pigeon d ’ un ton de mépris .
--------------------------------------------------------------------------------
SOURCE: "Can he be inconstant before being happy?"
TARGET: Serait-il volage avant d'être heureux?
PREDICTED: -- Serait - il heureux avant d ' être heureux ?
EPOCH: 11, CER: 0.6068885326385498, WER: 1.052022099494934, BLEU: 0.0
EPOCH: 11, Loss: 3.003523826599121
EPOCH: 12, Loss: 2.9121317863464355
--------------------------------------------------------------------------------
SOURCE: Pray make my excuses to Pratt for not keeping my engagement, and dancing with him to-night.
TARGET: « Dites a Pratt mon regret de ne pouvoir danser avec lui ce soir.
PREDICTED: - vous ma paye pour ne pas avoir de ma toilette et de la taille avec lui .
--------------------------------------------------------------------------------
SOURCE: But Gervaise, more curious, had not finished her questions.
TARGET: Mais Gervaise, plus curieuse, n’était pas au bout de ses questions.
PREDICTED: Mais Gervaise , plus curieux , n ’ avait pas fini de questions .
--------------------------------------------------------------------------------
SOURCE: The model, lying with her head thrown back and her torso twisted sometimes laughed and threw her bosom forward, stretching her arms.
TARGET: Dans le fond de l'atelier, un modèle, une femme était couchée, la tête ployée en arrière, le torse tordu, la hanche haute.
PREDICTED: Le cerveau , la tête tendu , la tête flottante , la battait parfois , et la chair , la serrait les bras .
--------------------------------------------------------------------------------
SOURCE: By the greatest accident, he did not fall off; from that moment he felt himself a hero.
TARGET: Par un grand hasard, il ne tomba pas, de ce moment il se sentit un héros.
PREDICTED: Par le plus grand malheur , il ne tomba pas de tomber , il se sentait un instant .
EPOCH: 13, CER: 0.6067250370979309, WER: 1.0736483335494995, BLEU: 0.0
EPOCH: 13, Loss: 2.8284356594085693
EPOCH: 14, Loss: 2.710888624191284
--------------------------------------------------------------------------------
SOURCE: Of the other men on board, I saw only my emotionless steward, who served me with his usual mute efficiency.
TARGET: Des gens du bord, je ne vis que l'impassible stewart, qui me servit avec son exactitude et son mutisme ordinaires.
PREDICTED: D ' autre part , je ne vis que mon stewart , qui me servait de son mouvement muet .
--------------------------------------------------------------------------------
SOURCE: "I shall," replied d’Artagnan, "and instantly."
TARGET: -- Je l'aurai, répondit d'Artagnan, et à l'instant même.
PREDICTED: -- Je le veux , répondit d ' Artagnan , et aussitôt .
--------------------------------------------------------------------------------
SOURCE: The other looked at him a moment, not surprised at what he said, but deeply moved at hearing him say it.
TARGET: L’autre le regarda un instant, non pas surpris de ce qu’il disait, mais profondément ému de le lui entendre dire.
PREDICTED: L ’ autre le regarda un moment , ne s ’ écria pas , mais il lui dit , en lui ayant profondément ému de dire .
--------------------------------------------------------------------------------
SOURCE: Then a workman, passing in the dim light, says in a low voice, bantering: "Don't you go, little girl. He'll do you a mischief!"
TARGET: Alors un ouvrier qui passe dans l’obscurité plaisante à mi-voix : – N’y va pas, ma petite, il te ferait mal !
PREDICTED: Alors , un ouvrier de la lumière passait dans la lumière , en passant , à la voix basse , il ne vous pas !
EPOCH: 15, CER: 0.6316837668418884, WER: 1.0773946046829224, BLEU: 0.0
EPOCH: 15, Loss: 2.585113525390625
EPOCH: 16, Loss: 2.4495458602905273
--------------------------------------------------------------------------------
SOURCE: She accompanied the guests into the arcade, and Laurent also went down with a lamp in his hand.
TARGET: Elle accompagna les invités jusque dans le passage, Laurent descendit aussi une lampe à la main.
PREDICTED: Elle accompagnait les invités , et Laurent se mettait en large , la lampe au passage , dans sa main .
--------------------------------------------------------------------------------
SOURCE: In your place I would stake the furniture against the horse."
TARGET: A votre place, je jouerais vos harnais contre votre cheval.
PREDICTED: Dans votre enjeu , je les meubles contre le cheval .
--------------------------------------------------------------------------------
SOURCE: – C’est mon mari, dit l’hôtesse.
TARGET: "This is my husband," said the landlady.
PREDICTED: " It is my husband ," said the landlady .
--------------------------------------------------------------------------------
SOURCE: I have prepared everything.
TARGET: J’ai tout préparé.
PREDICTED: J ’ ai tout réfléchi .
EPOCH: 17, CER: 0.6058925986289978, WER: 1.091953992843628, BLEU: 0.0
EPOCH: 17, Loss: 2.317978858947754
EPOCH: 18, Loss: 2.213956117630005
--------------------------------------------------------------------------------
SOURCE: As we moved forward, I heard a kind of pitter-patter above my head.
TARGET: Tout en avançant, j'entendais une sorte de grésillement au-dessus de ma tête.
PREDICTED: Nous en avant , j ' entendis un bon de ma tête .
--------------------------------------------------------------------------------
SOURCE: It doesn't matter; my little wood-house opens into the alley.
TARGET: Ça ne fait rien, mon petit bucher ouvre sur la ruelle…
PREDICTED: Ça ne donne pas l ' affaire , mon petit bois donne en grande allée .
--------------------------------------------------------------------------------
SOURCE: Already he was again blowing his horn, the band was lost in the distance, and the cry grew fainter:
TARGET: Déja, il s'était remis a souffler dans sa corne, la bande se perdait au loin, avec le cri affaibli:
PREDICTED: Déja , il soufflait de nouveau , la troupe s ' était perdue en loin , et le cri éclata .
--------------------------------------------------------------------------------
SOURCE: One hands the letter to the porter with a contrite air; profound melancholy in the gaze.
TARGET: On remet la lettre au portier d’un air contrit ; profonde mélancolie dans le regard.
PREDICTED: Une lettre , toute la lettre au portier d ' un air profondément profonde ; dans le regard .
EPOCH: 19, CER: 0.5947139263153076, WER: 1.0928906202316284, BLEU: 0.0
EPOCH: 19, Loss: 2.1283602714538574
EPOCH: 20, Loss: 2.0594046115875244
--------------------------------------------------------------------------------
SOURCE: "Do you forgive me, Jane?"
TARGET: -- Me pardonnez-vous? Jane.
PREDICTED: -- M . Jane me pardonne - vous , Jane ?
--------------------------------------------------------------------------------
SOURCE: "A disagreeable moment, a toll−gate, the passage of little to nothingness.
TARGET: Un mauvais moment, un péage, le passage de peu de chose à rien.
PREDICTED: – Un moment , un qui tient à la porte le centre du corps .
--------------------------------------------------------------------------------
SOURCE: If the earth opened beneath them a miracle would save them.
TARGET: Si la terre craquait sous eux, un miracle les sauverait.
PREDICTED: Si la terre les a sous un miracle , il les .
--------------------------------------------------------------------------------
SOURCE: Lydie drew back a few steps while he put his eye to a crack in the shutter.
TARGET: Lydie recula de quelques pas, pendant qu'il mettait un oeil a la fente du volet.
PREDICTED: Lydie recula quelques pas en , en lui disant ces quelques de prêtre .
EPOCH: 21, CER: 0.5940598249435425, WER: 1.0774797201156616, BLEU: 0.0
EPOCH: 21, Loss: 1.9962983131408691
EPOCH: 22, Loss: 1.941123604774475
--------------------------------------------------------------------------------
SOURCE: There he lodged a dozen of those pigeons which frequented the rocks of the plateau.
TARGET: On y logea une douzaine de ces pigeons qui fréquentaient les hauts rocs du plateau.
PREDICTED: Là il une douzaine de ces îles de récifs semblables .
--------------------------------------------------------------------------------
SOURCE: They even mention one oyster, about which I remain dubious, that supposedly contained at least 150 sharks."
TARGET: On a même cité une huître, mais je me permets d'en douter, qui ne contenait pas moins de cent cinquante requins.
PREDICTED: On même les uns , dont je suis , et les principes , qui contenait au moins de cinquante mètres .
--------------------------------------------------------------------------------
SOURCE: 'Your departure obliges me to speak ... It would be beyond my endurance not to see you any more.'
TARGET: « Votre départ m’oblige à parler… Il serait au-dessus de mes forces de ne plus vous voir. »
PREDICTED: Votre départ pour me parler … ce serait sans ma dernière fois de ne pas vous voir .
--------------------------------------------------------------------------------
SOURCE: Mr. Fogg and his two companions took their places on a bench opposite the desks of the magistrate and his clerk.
TARGET: Fogg, Mrs. Aouda et Passepartout s'assirent sur un banc en face des sièges réservés au magistrat et au greffier.
PREDICTED: Fogg et ses deux compagnons s ' sur un banc , s ' sur un banc du magistrat à son clerc et ses deux compagnons .
EPOCH: 23, CER: 0.5951004028320312, WER: 1.0983396768569946, BLEU: 0.0
EPOCH: 23, Loss: 1.8925286531448364
EPOCH: 24, Loss: 1.8515833616256714
--------------------------------------------------------------------------------
SOURCE: I have only been able to find a few which I seem to have jotted down almost unconsciously.
TARGET: Je n'ai plus retrouvé que quelques observations fugitives et prises machinalement pour ainsi dire.
PREDICTED: Je n ' ai pu seul trouver en un qui me semble sans m ' douter .
--------------------------------------------------------------------------------
SOURCE: The obstinate sailor did not reply, and let the conversation drop, quite determined to resume it again.
TARGET: L'entêté marin ne répondit pas et laissa tomber la conversation, bien décidé à la reprendre.
PREDICTED: L ' obstiné marin ne répondit pas , qui laissa la conversation se préparer à reprendre .
--------------------------------------------------------------------------------
SOURCE: "No, an islet lost in the Pacific, and which perhaps has never been visited."
TARGET: -- Non, un îlot perdu dans le Pacifique, et qui n'a jamais été visité peut-être!
PREDICTED: -- Non , un îlot du Pacifique , et qui n ' ait jamais été vu .
--------------------------------------------------------------------------------
SOURCE: "You couldn't get very hard hit over that." "Couldn't you?" he snarled.
TARGET: -- Vous n'avez pas dû être fortement atteint à ce jeu-là?
PREDICTED: -- Vous ne pouvez pas me rendre facilement dur de cette .
EPOCH: 25, CER: 0.5957693457603455, WER: 1.0738186836242676, BLEU: 0.0
EPOCH: 25, Loss: 1.8123170137405396
EPOCH: 26, Loss: 1.7772434949874878
--------------------------------------------------------------------------------
SOURCE: C’était un homme tres rusé que ce Stangerson et qui se tenait toujours sur ses gardes.
TARGET: He was cunning, was Stangerson, and always on his guard.
PREDICTED: It was a man who was up with this and passed them on his oaths .
--------------------------------------------------------------------------------
SOURCE: 'Look at that!
TARGET: – Allons, bon !
PREDICTED: – Voyez !
--------------------------------------------------------------------------------
SOURCE: – Il ne commettra plus la faute de passer par-dessus les murs, dit Clélia ; mais il sortira par la porte, s’il est acquitté.
TARGET: "He will not make the mistake of going over the walls again," said Clelia, "but he will leave by the door if he is acquitted."
PREDICTED: " He will not come more the time to stop but the ," said Clelia . " but he is from the door .
--------------------------------------------------------------------------------
SOURCE: "Well, I'm going to chuck him out," replied Joe.
TARGET: – Ma foi, je vais le flanquer dehors.
PREDICTED: -- Eh bien , je vais le gronder , répliqua Joe .
EPOCH: 27, CER: 0.5889015793800354, WER: 1.0851426124572754, BLEU: 0.0
EPOCH: 27, Loss: 1.7460441589355469
EPOCH: 28, Loss: 1.718700885772705
--------------------------------------------------------------------------------
SOURCE: She turned down a street; she recognised him by his curling hair that escaped from beneath his hat.
TARGET: Elle tournait une rue; elle le reconnaissait à sa chevelure frisée qui s’échappait de son chapeau.
PREDICTED: Elle se retourna sur une rue , elle le reconnut en relevant ses cheveux qui s ’ échappa du chapeau de son chapeau .
--------------------------------------------------------------------------------
SOURCE: Hunger, the fresh air, the calm quiet weather, after the commotions we had gone through, all contributed to give me a good appetite.
TARGET: Le besoin, le grand air, le calme après les agitations, tout contribuait à me mettre en appétit.
PREDICTED: La faim , le temps est nouveau , le calme du temps présent , après l ' être , tous mes chances pour m ' établir .
--------------------------------------------------------------------------------
SOURCE: 'Your departure obliges me to speak ... It would be beyond my endurance not to see you any more.'
TARGET: « Votre départ m’oblige à parler… Il serait au-dessus de mes forces de ne plus vous voir. »
PREDICTED: Votre départ m ’ est accoutumé à parler … Il sera possible de ne pas vous revoir davantage .
--------------------------------------------------------------------------------
SOURCE: Born a Gascon but bred a Norman, he grafted upon his southern volubility the cunning of the Cauchois.
TARGET: Né Gascon, mais devenu Normand, il doublait sa faconde méridionale de cautèle cauchoise.
PREDICTED: Il faut un vrai Gascon mais un du Sud , et il fallait .
EPOCH: 29, CER: 0.5854974389076233, WER: 1.06956148147583, BLEU: 0.0
EPOCH: 29, Loss: 1.6938942670822144
EPOCH: 30, Loss: 1.6710034608840942
--------------------------------------------------------------------------------
SOURCE: The day was drawing in.
TARGET: Le jour tombait.
PREDICTED: Le jour fut dans le salon .
--------------------------------------------------------------------------------
SOURCE: "Come at once," she said; "they are laying the table, and we'll have supper."
TARGET: --Venez vite, on met la table, disait-elle, nous allons souper.
PREDICTED: « Venez aussitôt ,» dit - elle . Elles me la table , et nous allons souper .
--------------------------------------------------------------------------------
SOURCE: She does not yet leave her dressing-room.
TARGET: Elle sera satisfaite de vous voir tous les trois.
PREDICTED: Elle ne quitta pas son cabinet de toilette .
--------------------------------------------------------------------------------
SOURCE: "Levaque's wife is catching it," Maheu peacefully stated as he scraped the bottom of his bowl with the spoon.
TARGET: —La Levaque reçoit sa danse, constata paisiblement Maheu, en train de racler le fond de sa jatte avec la cuiller.
PREDICTED: — La Levaque de — Dame est enceinte , dit paisiblement Maheu , en s ' asseyant sur ses .
EPOCH: 31, CER: 0.5903732776641846, WER: 1.0876117944717407, BLEU: 0.0
EPOCH: 31, Loss: 1.650494933128357
EPOCH: 32, Loss: 1.632812738418579
--------------------------------------------------------------------------------
SOURCE: A heavy bag immediately plunged into the sea.
TARGET: Un sac pesant tomba aussitôt à la mer.
PREDICTED: Un sac se précipité aussitôt à la mer .
--------------------------------------------------------------------------------
SOURCE: Ah! your dress is damp."
TARGET: Ah! ta robe est mouillée!
PREDICTED: Ah ! ta robe est humide !
--------------------------------------------------------------------------------
SOURCE: "Down there?" repeated my uncle.
TARGET: --Là-bas?» répond mon oncle.
PREDICTED: -- À bas ? répétait mon oncle .
--------------------------------------------------------------------------------
SOURCE: He entangled himself in between four chairs all at once.
TARGET: Il s’embarrassait dans quatre chaises à la fois.
PREDICTED: Il s ’ embarrassait dans quatre chaises .
EPOCH: 33, CER: 0.5885300040245056, WER: 1.0750106573104858, BLEU: 0.0
EPOCH: 33, Loss: 1.6164131164550781
EPOCH: 34, Loss: 1.6022331714630127
--------------------------------------------------------------------------------
SOURCE: "There is no one with a better heart than Charles; but his own life moves so smoothly that he cannot understand that others may have trouble.
TARGET: Personne n'a meilleur coeur que Charles, mais sa vie s'écoule si doucement qu'il ne peut comprendre que d'autres aient des ennuis.
PREDICTED: Il n ’ y a personne pour le cœur mieux que Charles ; mais sa vie ne pas qu ’ il puisse comprendre les autres .
--------------------------------------------------------------------------------
SOURCE: "Musnier, we'll kiss your wife."
TARGET: – Musnier, nous chiffonnerons ta femme.
PREDICTED: – Musnier , nous ton femme .
--------------------------------------------------------------------------------
SOURCE: It consisted of these words:
TARGET: Cette dédicace portrait ces seul mots:
PREDICTED: Il se présenta à ces mots .
--------------------------------------------------------------------------------
SOURCE: No weakness!
TARGET: Pas de faiblesse!
PREDICTED: Non ! pas !
EPOCH: 35, CER: 0.5852893590927124, WER: 1.0850574970245361, BLEU: 0.0
EPOCH: 35, Loss: 1.5884604454040527
EPOCH: 36, Loss: 1.5768758058547974
--------------------------------------------------------------------------------
SOURCE: Demandez à votre baron de quelle peine il veut punir ce moment de folie.
TARGET: Ask your Barone with what penalty he proposes to punish this moment of folly?"
PREDICTED: to your mind if he had the moment to have only loved her .
--------------------------------------------------------------------------------
SOURCE: "And I," said Treville, coldly, "I have some pretty things to tell your Majesty concerning these gownsmen."
TARGET: -- Et moi, dit froidement M. de Tréville, j'en ai de belles à apprendre à Votre Majesté sur ses gens de robe.
PREDICTED: -- Et moi , dit Tréville froidement , j ' ai là des choses à Votre Majesté .
--------------------------------------------------------------------------------
SOURCE: Chuck him in among his own cinders!
TARGET: Roulez-le dans son tas de cendre.
PREDICTED: - le parmi les sienne !
--------------------------------------------------------------------------------
SOURCE: The sun, fairly low on the horizon, struck full force on the houses in this town, accenting their whiteness.
TARGET: Le soleil, assez bas sur l'horizon, frappait en plein les maisons de la ville et faisait ressortir leur blancheur.
PREDICTED: La brise , belle à l ' horizon , toute de bonne humeur vers les maisons , leur souffle .
EPOCH: 37, CER: 0.5861069560050964, WER: 1.0811408758163452, BLEU: 0.0
EPOCH: 37, Loss: 1.566548228263855
EPOCH: 38, Loss: 1.5586274862289429
--------------------------------------------------------------------------------
SOURCE: Guillaume! thou art the largest, and Pasquier is the smallest, and Pasquier does best.
TARGET: Rends-les tous sourds comme moi.
PREDICTED: Guillaume ! tu es l ’ art le plus grand et le plus gros , et Pasquier est le plus grand .
--------------------------------------------------------------------------------
SOURCE: It was empty.
TARGET: Il était vide!
PREDICTED: C ' était vide .
--------------------------------------------------------------------------------
SOURCE: "Well, Herbert," replied the engineer, "you are right to attach great importance to this fact.
TARGET: «Bien, Harbert, répondit l'ingénieur, tu as raison d'attacher une grande importance à ce fait.
PREDICTED: -- Eh bien , Harbert , répondit l ' ingénieur , vous devez prudent de résoudre en cet endroit .
--------------------------------------------------------------------------------
SOURCE: Bingley was every thing that was charming, except the professed lover of her daughter.
TARGET: Bennet ne réussirent pas ce soir-la.
PREDICTED: Jones si la chose était charmante , c ’ était l ’ amant de sa fille .
EPOCH: 39, CER: 0.5860028862953186, WER: 1.0824180841445923, BLEU: 0.0
EPOCH: 39, Loss: 1.5520380735397339
```
