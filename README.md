# Introdução à inteligência artificial aplicada à veículos autônomos
Uma introdução aos assuntos que serão desenvolvidos na disciplina de Sistemas Autônomos.

## Universidade de Fortaleza - Unifor
### Sistemas Autônomos

**Professor: Afonso Henrique**

**Autor: Marcos Vinícius Cândido Leitão**

**Fortaleza - 2017**

### Introdução

  Neste relatório irei apresentar alguns dos assuntos desenvolvidos durante a disciplina de Sistemas Autônomos. Inicialmente irei demonstrar uma aplicação envolvendo o software OpenAi, e em seguida irei apresentar algumas análises importantes sobre o processamento de imagens do Carro Autônomo desenvolvido pela NVIDIA.


## -OpenAi Universe

  #### Softwares

  Inicialmente é necessário o download do Anaconda. Podendo acessar através deste link: https://www.continuum.io/downloads
  
  <img width="600" alt="captura de tela 2017-12-06 as 16 53 36" src="https://user-images.githubusercontent.com/31712391/33682526-7e2ab1cc-daa6-11e7-95cf-ff05f4966271.png">
  
  Sendo possível baixar para vários sistemas operacionais.

  Logo após será necessário a instalação do Docker que pode ser baixado para Mac ou Windows através deste link: https://store.docker.com/editions/community/docker-ce-desktop-mac
  
  <img width="600" alt="captura de tela 2017-12-06 as 17 09 14" src="https://user-images.githubusercontent.com/31712391/33683053-61388786-daa8-11e7-8e9f-37a63d88aff6.png">
  
  #### Implementação
  
  Antes de continuar devemos verificar se o Docker está ativo.

<img width="300" alt="captura de tela 2017-12-06 as 17 48 12" src="https://user-images.githubusercontent.com/31712391/33684692-de46c936-daad-11e7-83bd-eb422b8b0708.png">

  Após verificar o Docker iremos criar um Environment, então você irá no Anaconda e cria-lo, no meu caso criei com o nome de "open-ai"
  
  <img width="600" alt="captura de tela 2017-12-06 as 17 26 44" src="https://user-images.githubusercontent.com/31712391/33683892-2bbca51c-daab-11e7-9d0b-6c73870728d0.png">
  
  Agora iremos fazer um clone de um repositório do git usando as linhas de comando abaixo no terminal do Environment:
 

      cd ~/Downloads
  
      git clone https://github.com/openai/universe.git
  
      cd universe
  
      sudo -H "PATH=$PATH" pip install -e .'
  
  Devemos por precaução verificar se o Docker realmente está funcionando, digitando no 'docker ps' e caso esteja normal irá aparecer no terminal como é demonstrado abaixo:
  
  <img width="571" alt="captura de tela 2017-12-06 as 18 01 42" src="https://user-images.githubusercontent.com/31712391/33685189-8d295c10-daaf-11e7-89d7-f170e5c1cbb3.png">

  Também devemos verificar se o python está corretamente instalado digitando as linhas abaixo:
  
      python
      import universe
      
  Caso esteja tudo corretamente funcionando deverá aparecer no terminal as linhas de codigos abaixo
  
  <img width="571" alt="captura de tela 2017-12-06 as 18 04 48" src="https://user-images.githubusercontent.com/31712391/33685326-fab9bf04-daaf-11e7-9a3e-b2fc3de3ab4d.png">
  
  Para finalizar, devemos criar um documento com o nome 'run.py' com as linhas de codigo abaixo:
  
      import gym
      import universe # register the universe environments
      env = gym.make('flashgames.DuskDrive-v0')
      env.configure(remotes=1) # automatically creates a local docker container
      observation_n = env.reset()
      while True:
        action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n] # your agent here
        observation_n, reward_n, done_n, info = env.step(action_n)
        env.render()
 
  Após isso devemos digitar a linha de cógido abaixo para exetutar o programa:
  
        python run.py
        
  Se funcionar irá iniciar um download do conteúdo para o funcionamento do jogo como demonstrado na tela abaixo:
  
 <img width="1001" alt="captura de tela 2017-12-06 as 18 48 16" src="https://user-images.githubusercontent.com/31712391/33687737-ce7ab4cc-dab7-11e7-8056-e8c6b296e503.png">
 
  Finalizando o download irá iniciar o jogo e o sistema irá começar a realizar o aprendizado baseado nas tentativas dentro de um sistema de recompensas do jogo.
  
  <img width="766" alt="captura de tela 2017-12-06 as 19 03 29" src="https://user-images.githubusercontent.com/31712391/33687822-2f5299b8-dab8-11e7-960a-c657fb41ed7c.png">




  
  







