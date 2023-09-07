# Session 15

## Introduction

This assignment focuses on understanding the transformer code by converting the translation transformer (encoder + decoder) from vanilla pytorch to pytorch lightning.

### Target
1. Port training code to Pytorch Lightning
2. Achieve Loss < 4

### Metrics
Final Train Loss: 3.392
Model trained for 11 epochs

## Data Exploration

The dataset is the Opus Books dataset, involving translation from English to Italian.

```
SOURCE: CHAPTER XX — ARE FORTRESSES, AND MANY OTHER THINGS TO WHICH PRINCES OFTEN RESORT, ADVANTAGEOUS OR HURTFUL?
TARGET: Cap.20 An arces et multa alia quae cotidie a principibus fiunt utilia an inutilia sint. [Se le fortezze e molte altre cose, che ogni giorno si fanno da’ principi, sono utili o no]

SOURCE: We had a boat at our stern just before the storm, but she was first staved by dashing against the ship’s rudder, and in the next place she broke away, and either sunk or was driven off to sea; so there was no hope from her.
TARGET: Prima della burrasca avevamo a poppa una scialuppa, ma sfondatasi contro al timone e infrantesi

SOURCE: 'Oh no, Masha, Mr. Levin only says he can't believe...' said Kitty, blushing for Levin, and Levin understanding this became still more irritated and wished to answer, but Vronsky, with his bright and frank smile, came at once to the rescue of the conversation, which was threatening to become unpleasant.
TARGET: — Ma no, Maša, Konstantin Dmitric dice che non ci può credere — disse Kitty, arrossendo per Levin, e questi lo capì e, irritatosi ancor più, voleva rispondere; ma Vronskij col suo sorriso aperto, cordiale, venne subito in aiuto della conversazione che minacciava di farsi spiacevole.

SOURCE: And if it be urged that whoever is armed will act in the same way, whether mercenary or not, I reply that when arms have to be resorted to, either by a prince or a republic, then the prince ought to go in person and perform the duty of a captain; the republic has to send its citizens, and when one is sent who does not turn out satisfactorily, it ought to recall him, and when one is worthy, to hold him by the laws so that he does not leave the command.
TARGET: E se si responde che qualunque arà le arme in mano farà questo, o mercennario o no, replicherei come l’arme hanno ad essere operate o da uno principe o da una repubblica. El principe debbe andare in persona, e fare lui l'offizio del capitano; la repubblica ha a mandare sua cittadini; e quando ne manda uno che non riesca valente uomo, debbe cambiarlo; e quando sia, tenerlo con le leggi che non passi el segno.
```

## Learning Curves

<img width="417" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/fde97cad-0945-4c27-b0a8-6fd4cd6093fa">

<img width="411" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/e34865ad-80f1-44cb-be85-cd8253fa099e">


## Training Log

```
EPOCH:0
--------------------------------------------------------------------------------
SOURCE: CHAPTER XX — ARE FORTRESSES, AND MANY OTHER THINGS TO WHICH PRINCES OFTEN RESORT, ADVANTAGEOUS OR HURTFUL?
TARGET: Cap.20 An arces et multa alia quae cotidie a principibus fiunt utilia an inutilia sint. [Se le fortezze e molte altre cose, che ogni giorno si fanno da’ principi, sono utili o no]
PREDICTED: Dopo il mio , e , , e , , e , , , .
--------------------------------------------------------------------------------
SOURCE: We had a boat at our stern just before the storm, but she was first staved by dashing against the ship’s rudder, and in the next place she broke away, and either sunk or was driven off to sea; so there was no hope from her.
TARGET: Prima della burrasca avevamo a poppa una scialuppa, ma sfondatasi contro al timone e infrantesi le corde che la teneano, andò a sommergersi o il mare la trascinò lontano da noi.
PREDICTED: il mio giorno , ma il mio giorno , ma non si , e non si , e , e non si a , e non si a , e non si .
--------------------------------------------------------------------------------
SOURCE: 'Oh no, Masha, Mr. Levin only says he can't believe...' said Kitty, blushing for Levin, and Levin understanding this became still more irritated and wished to answer, but Vronsky, with his bright and frank smile, came at once to the rescue of the conversation, which was threatening to become unpleasant.
TARGET: — Ma no, Maša, Konstantin Dmitric dice che non ci può credere — disse Kitty, arrossendo per Levin, e questi lo capì e, irritatosi ancor più, voleva rispondere; ma Vronskij col suo sorriso aperto, cordiale, venne subito in aiuto della conversazione che minacciava di farsi spiacevole.
PREDICTED: — No , non è nulla — disse Levin , ma Levin , ma Levin , e Levin , e Levin , e Levin , e Levin , e Levin , e Levin , e Levin , e Levin si mise a Levin .
--------------------------------------------------------------------------------
SOURCE: And if it be urged that whoever is armed will act in the same way, whether mercenary or not, I reply that when arms have to be resorted to, either by a prince or a republic, then the prince ought to go in person and perform the duty of a captain; the republic has to send its citizens, and when one is sent who does not turn out satisfactorily, it ought to recall him, and when one is worthy, to hold him by the laws so that he does not leave the command.
TARGET: E se si responde che qualunque arà le arme in mano farà questo, o mercennario o no, replicherei come l’arme hanno ad essere operate o da uno principe o da una repubblica. El principe debbe andare in persona, e fare lui l'offizio del capitano; la repubblica ha a mandare sua cittadini; e quando ne manda uno che non riesca valente uomo, debbe cambiarlo; e quando sia, tenerlo con le leggi che non passi el segno.
PREDICTED: E che non è che , e ' uomini , non è vero , e non è vero , e non è vero , e non si , e non si , e non si , e non si , e ' uomini , e ' uomini , e non si , e non si , e non si , e ' uomini , e non si , e non si , e non si , e ' uomini , e non si , e non si .

EPOCH:2
--------------------------------------------------------------------------------
SOURCE: 'So I heard.'
TARGET: — Sì, ho sentito.
PREDICTED: — Allora io .
--------------------------------------------------------------------------------
SOURCE: It is not a personal affair of my own but one of public welfare.
TARGET: Questa faccenda non riguarda solo la mia persona, ma qui si tratta del bene generale.
PREDICTED: Non è una cosa simile a me , ma una cosa di Dio .
--------------------------------------------------------------------------------
SOURCE: On one of the staircases, I met the physician of the family.
TARGET: Su una delle scale incontrai il medico della famiglia.
PREDICTED: Dopo un uomo , mi , mi parve che il mondo .
--------------------------------------------------------------------------------
SOURCE: 'No, I have done with that; it is time for me to die.'
TARGET: — No, per me è finita. È tempo di morire.
PREDICTED: — No , io sono content .

EPOCH:4
--------------------------------------------------------------------------------
SOURCE: 'No, I have done with that; it is time for me to die.'
TARGET: — No, per me è finita. È tempo di morire.
PREDICTED: — No , io ho fatto con me ; è per me per me .
--------------------------------------------------------------------------------
SOURCE: And the heroism of the Serbs and Montenegrins, fighting for a great cause, aroused in the whole nation a desire to help their brothers not only with words but by deeds.
TARGET: E l’eroismo dei serbi e dei montenegrini, che lottavano per una grande causa, aveva generato in tutto il popolo il desiderio di aiutare i fratelli non più con la parola, ma con l’azione.
PREDICTED: E le della e le , per la guerra , si in modo , non solo non si a non avere né la sua volontà , ma la sua volontà non si .
--------------------------------------------------------------------------------
SOURCE: 'Oh no, Masha, Mr. Levin only says he can't believe...' said Kitty, blushing for Levin, and Levin understanding this became still more irritated and wished to answer, but Vronsky, with his bright and frank smile, came at once to the rescue of the conversation, which was threatening to become unpleasant.
TARGET: — Ma no, Maša, Konstantin Dmitric dice che non ci può credere — disse Kitty, arrossendo per Levin, e questi lo capì e, irritatosi ancor più, voleva rispondere; ma Vronskij col suo sorriso aperto, cordiale, venne subito in aiuto della conversazione che minacciava di farsi spiacevole.
PREDICTED: — Oh , no , principessa , Konstantin Dmitric , non può essere impossibile — disse Kitty , arrossendo per un ’ altra volta , e Levin , senza dubbio di nuovo aver pensato a Vronskij , e , dopo aver notato che il suo sorriso era stato contento di lei , era stato proprio proprio in quel momento in cui era stato stato stato deciso di lei .
--------------------------------------------------------------------------------
SOURCE: The pain she had inflicted on herself and her husband would now, she thought, be compensated for by the fact that the matter would be settled.
TARGET: Il dolore ch’ella aveva causato a se stessa e al marito nel pronunziare quelle parole, sarebbe stato compensato, così ella immaginava, dal fatto che tutto si sarebbe definito.
PREDICTED: La gelosia , per , si era per lei e ora , per quanto sarebbe stato necessario , sarebbe stato necessario per il denaro , sarebbe stato necessario .

EPOCH:6
--------------------------------------------------------------------------------
SOURCE: One who did not conquer was Giovanni Acuto, and since he did not conquer his fidelity cannot be proved; but every one will acknowledge that, had he conquered, the Florentines would have stood at his discretion.
TARGET: Quello che non vinse fu Giovanni Aucut, del quale, non vincendo, non si poteva conoscere la fede; ma ognuno confesserà che, vincendo, stavano Fiorentini a sua discrezione.
PREDICTED: Una che non si el principe , e , avendo , avendo , non si può non essere odiato ; ma , che li aveva , li Orsini , li Orsini , li Orsini li Orsini , si .
--------------------------------------------------------------------------------
SOURCE: One who did not conquer was Giovanni Acuto, and since he did not conquer his fidelity cannot be proved; but every one will acknowledge that, had he conquered, the Florentines would have stood at his discretion.
TARGET: Quello che non vinse fu Giovanni Aucut, del quale, non vincendo, non si poteva conoscere la fede; ma ognuno confesserà che, vincendo, stavano Fiorentini a sua discrezione.
PREDICTED: Una che non si el principe , e , avendo , avendo , non si può non essere odiato ; ma , che li aveva , li Orsini , li Orsini , li Orsini li Orsini , si .
--------------------------------------------------------------------------------
SOURCE: Wishing to show his independence and to get promotion, he had refused a post that was offered him, hoping that this refusal would enhance his value, but it turned out that he had been too bold and he was passed over. Having then perforce to assume the role of an independent character, he played it very adroitly and cleverly, as though he had no grudge against anyone, did not feel himself at all offended, and only wished to be left in peace to enjoy himself.
TARGET: Per dar prova della propria indipendenza e di voler progredire, aveva rifiutato una posizione offertagli, sperando che questo rifiuto potesse conferirgli maggior prestigio; accadde invece che fu giudicato troppo temerario, e fu lasciato stare; e ora, volente o nolente, acquistatasi questa fama di uomo libero, cercava di sostenerla, comportandosi con finezza e intelligenza, in modo da parere che non avesse rancore contro nessuno, che non si considerasse offeso da nessuno, e che volesse solo starsene in pace, perché contento di sé.
PREDICTED: per parlargli del suo posto , voleva dare un ’ occhiata , ma che lo tormentava , per questo , per questo , il quale egli avrebbe dovuto , ma che era stato molto sicuro che lui era stato fatto molto , e che lui era stato fatto un ’ influenza , sebbene egli non avesse avuto parte di sé , e non solo non poteva capire che si con lui stesso , e che non era necessario , e non si poteva capire che cosa con lui .
--------------------------------------------------------------------------------
SOURCE: And the heroism of the Serbs and Montenegrins, fighting for a great cause, aroused in the whole nation a desire to help their brothers not only with words but by deeds.
TARGET: E l’eroismo dei serbi e dei montenegrini, che lottavano per una grande causa, aveva generato in tutto il popolo il desiderio di aiutare i fratelli non più con la parola, ma con l’azione.
PREDICTED: E i e i , per esempio , per un po ’ di guerra , si misero a dare uno scopo , ma non solo con la loro volontà , ma con la forza di non essere odiato .

EPOCH:8
--------------------------------------------------------------------------------
SOURCE: I heard one of your kind an hour ago, singing high over the wood: but its song had no music for me, any more than the rising sun had rays.
TARGET: Da un'ora sento le vostre sorelle cantare nel bosco, ma per me il loro canto non aveva armonia:
PREDICTED: " In un ' ora vi sentii udire un ' ora , ma il rumore delle sue finestre mi avevano fatto udire la musica , più alta e più alto che aveva una carnagione bianca .
--------------------------------------------------------------------------------
SOURCE: 'I had one girl, but God released me. I buried her in Lent.'
TARGET: — Ho avuto una bambina, ma Dio mi ha liberata, l’ho sotterrata a quaresima.
PREDICTED: — Ho una ragazza , ma io mi la sua parola .
--------------------------------------------------------------------------------
SOURCE: So I lay, and so he stood.
TARGET: Ero sdraiato così io, così in piedi stava lui.
PREDICTED: Così mi trovai , e così mi parve .
--------------------------------------------------------------------------------
SOURCE: We had here the Word of God to read, and no farther off from His Spirit to instruct than if we had been in England.
TARGET: Qui avevamo per leggerli i divini volumi, nè lo spirito del Signore era per istruirci più lontano da noi che nol sarebbe stato nell’Inghilterra.
PREDICTED: Noi avevamo la sua superiorità , e non avevamo più voglia di di , se avessimo avuto l ’ Inghilterra .
```
