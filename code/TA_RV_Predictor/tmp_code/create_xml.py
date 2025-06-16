import xml.dom.minidom



def CreateXML():
  #在内存中创建一个空的文档
  doc = xml.dom.minidom.Document() 
  #创建一个根节点Managers对象
  root = doc.createElement('template') 
  #设置根节点的属性
  #将根节点添加到文档对象中
  doc.appendChild(root) 

  node_name = doc.createElement('name')
  node_name.appendChild(doc.createTextNode('a_to_b'))

  node_declaration = doc.createElement('declaration')
  node_declaration.appendChild(doc.createTextNode('here we need to know something'))

  root.appendChild(node_name)
  root.appendChild(node_declaration)

  LocationList = ['id1', 'id2', 'id3']

  TrasitionList = [{'from' : 'id1',  'to' : 'id2', 'action' : 'a', 'time' : 'x > 30'},
                {'from' : 'id1',  'to' : 'id3', 'action' : 'b', 'time' : 'x < 10'},
                {'from' : 'id2',  'to' : 'id3', 'action' : 'a', 'time' : 'x > 15'}
  ]



  for i in LocationList:
    node_location = doc.createElement('location')
    node_location.appendChild(doc.createTextNode(str(i)))
    
    root.appendChild(node_location)

  for j in TrasitionList:
    node_trasition = doc.createElement('trasition')
    node_from = doc.createElement('from')
    #
    node_from.setAttribute('x','5')
    node_from.setAttribute('y','5')
    #给叶子节点name设置一个文本节点，用于显示文本内容
    node_from.appendChild(doc.createTextNode(str(j['from'])))

    node_to = doc.createElement("to")
    node_to.appendChild(doc.createTextNode(str(j["to"])))

    node_action = doc.createElement("action")
    node_action.appendChild(doc.createTextNode(str(j["action"])))

    node_time = doc.createElement("time")
    node_time.appendChild(doc.createTextNode(str(j["time"])))

    #将各叶子节点添加到父节点node_trasition中，
    node_trasition.appendChild(node_from)
    node_trasition.appendChild(node_to)
    node_trasition.appendChild(node_action)
    node_trasition.appendChild(node_time)
    
    #最后将node_trasition添加到根节点root中
    root.appendChild(node_trasition)

    
  #开始写xml文档
  fp = open('FSM.xml', 'w')
  doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
  print('write down successfully')