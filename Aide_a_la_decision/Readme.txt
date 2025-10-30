Algorithme de Mariage Stable = "gale shapley"

1. Implanter un programme pour générer des préférences aléatoires des étudiants et des établissements dans un csv.
	Création d'un script python "preferences.py" qui génère les préférences.
		Dans le fichier généré, il y a 5 points importants :
			-"n" -> taille de l'ensemble (que l'on pourra changer pour les tests)
			-"students" -> pour chaque étudiants, on a ses préférences pour les écoles
			-"schools" -> pour chaque écoles, on a ses préférences pour les étudiants
			-"rank_students -> même principes que students, mais permet d'aller chercher plus rapidement (sera utile en cas de changement de préférences)
			-"rank_schools -> même principes que schools, mais permet d'aller chercher plus rapidement (sera utile en cas de changement de préférences)
			
	Commande d'exécution : python3 preferences.py --n x --seed y --out /path/to/prefs.json
		-"python3" -> c'est la version que j'utilise
		-"preferences.py" -> nom du script
		-"--n x"	-> création pour x étudiants et x établissements
		-"--seed y"	-> permet de tester l'aléatoire, si seed=123, alors on aura toujours le même fichier	
					   on peut enlever seed pour un résultat 100% aléatoire
					   nous sera utile pour tester nos résultats
		-"--out /path/to/prefs.json"	-> l'endroit ou sortira le fichier des préférences

Pour cette partie, on a réussi sans difficulté a créé un script python qui nous génère un csv contenant x nombre d'étudiants et d'écoles avec leur priorité d'affectation.

2. Implanter l’algorithme du mariage stable
	Création d'un script python "gale_shapley" qui nous génère les résultats dans un scv
		-Prends en entré le csv généré grâce à la partie 1.
		-proposants = étudiants
		-receveurs = écoles
		-fonctionnement :
			-Tout est libre au départ
			-Tant qu'il reste un proposant de libre :
				-Il propose à son prochain receveur préféré qu’il n’a pas encore essayé.
				-Si le receveur est libre -> il accepte temporairement.
				-Sinon, le receveur compare le nouveau proposant avec celui déjà accepté :
					-s’il préfère le nouveau -> il remplace et rejette l’ancien,
					-sinon, il garde l’actuel.
			-Quand plus aucun proposant n’est libre, l’appariement est stable.
			
		Point important sur DA_E et DA_S :
			-Indique l'ordre des propositions
			-Dans notre cas, DA_E = étudiants qui sont utilisés en premiers
			-Dans le cas du DA_S = écoles qui sont priorisés
			
	Commande d'exécution : python3 gale_shapley.py --in prefs_123.json --mode DA_E --out match1.json
		-"gale_shapley.py" -> nom du script
		-"--in prefs_123" -> les entrées que l'on utilise (123 car c'est la seed de génération aléatoire)
		-"mode DA_E" -> on prend les étudiants comme priorité dans l'algorithme
		-"--out match_DAE_123.json" -> l'endroit ou se créera le fichier (on précise le jeu de donnée, ainsi que le mode)
		
Pour cette partie, on a réussi à utiliser le csv de la question précédente afin de résoudre le problème. On a construit l'algorithme de "Mariage Stable".
On a une zone d'action sur la génération aléatoire ( n + seed), ainsi que sur l'importance des proposants/receveurs (mode DA_E/S).
