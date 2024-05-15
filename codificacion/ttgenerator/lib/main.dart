import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        // This is the theme of your application.
        //
        // TRY THIS: Try running your application with "flutter run". You'll see
        // the application has a blue toolbar. Then, without quitting the app,
        // try changing the seedColor in the colorScheme below to Colors.green
        // and then invoke "hot reload" (save your changes or press the "hot
        // reload" button in a Flutter-supported IDE, or press "r" if you used
        // the command line to start the app).
        //
        // Notice that the counter didn't reset back to zero; the application
        // state is not lost during the reload. To reset the state, use hot
        // restart instead.
        //
        // This works for code too, not just values: Most code changes can be
        // tested with just a hot reload.
        scaffoldBackgroundColor: const Color(0xFFDDDDDD),
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.black),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  // This widget is the home page of your application. It is stateful, meaning
  // that it has a State object (defined below) that contains fields that affect
  // how it looks.

  // This class is the configuration for the state. It holds the values (in this
  // case the title) provided by the parent (in this case the App widget) and
  // used by the build method of the State. Fields in a Widget subclass are
  // always marked "final".

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  @override
  Widget build(BuildContext context) {
    // This method is rerun every time setState is called, for instance as done
    // by the _incrementCounter method above.
    //
    // The Flutter framework has been optimized to make rerunning build methods
    // fast, so that you can just rebuild anything that needs updating rather
    // than having to individually change instances of widgets.
    return Scaffold(
      body: Row(
        children: [
          Expanded(
            flex: 2,
            child: Container(
              color: Colors.white,
              child: const Center(
                child: Text('Lienzo'),
              ),
            ),
          ),
          Container(
            width: 1,
            color: Colors.black,
          ),
          const Expanded(
            flex: 1,
            child: SingleChildScrollView(
              child: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text('Parametros'),
                    Divider(
                      color: Colors.black,
                    ),
                    Text('Tamaño de Lienzo'),
                    SizedBox(
                      height: 100,
                      width: 180,
                      child: TextField(
                        decoration: InputDecoration(
                            labelText: 'Altura',
                            labelStyle: TextStyle(color: Colors.black),
                            border: OutlineInputBorder(),
                            focusedBorder: OutlineInputBorder(
                              borderSide: BorderSide(color: Colors.black),
                            )),
                      ),
                    ),
                    SizedBox(
                      height: 100,
                      width: 180,
                      child: TextField(
                        decoration: InputDecoration(
                            labelText: 'Ancho',
                            labelStyle: TextStyle(color: Colors.black),
                            border: OutlineInputBorder(),
                            focusedBorder: OutlineInputBorder(
                              borderSide: BorderSide(color: Colors.black),
                            )),
                      ),
                    ),
                    Divider(
                      color: Colors.black,
                    ),
                    Text('Configuracion de la maquina'),
                    SizedBox(
                      height: 100,
                      width: 200,
                      child: TextField(
                        decoration: InputDecoration(
                            labelText: 'Velocidad de Avance',
                            labelStyle: TextStyle(color: Colors.black),
                            border: OutlineInputBorder(),
                            focusedBorder: OutlineInputBorder(
                              borderSide: BorderSide(color: Colors.black),
                            )),
                      ),
                    ),
                    SizedBox(
                      height: 100,
                      width: 180,
                      child: TextField(
                        decoration: InputDecoration(
                            labelText: 'Amperaje',
                            labelStyle: TextStyle(color: Colors.black),
                            border: OutlineInputBorder(),
                            focusedBorder: OutlineInputBorder(
                              borderSide: BorderSide(color: Colors.black),
                            )),
                      ),
                    ),
                    SizedBox(
                      height: 100,
                      width: 180,
                      child: TextField(
                        decoration: InputDecoration(
                            labelText: 'Diametro',
                            labelStyle: TextStyle(color: Colors.black),
                            border: OutlineInputBorder(),
                            focusedBorder: OutlineInputBorder(
                              borderSide: BorderSide(color: Colors.black),
                            )),
                      ),
                    ),
                    SizedBox(
                      height: 100,
                      width: 180,
                      child: TextField(
                        decoration: InputDecoration(
                            labelText: 'Direccion',
                            labelStyle: TextStyle(color: Colors.black),
                            border: OutlineInputBorder(),
                            focusedBorder: OutlineInputBorder(
                              borderSide: BorderSide(color: Colors.black),
                            )),
                      ),
                    ),
                    Divider(
                      color: Colors.black,
                    ),
                    SizedBox(
                      width: 374,
                      height: 150,

                    )
                  ],
                ),
              ),
            ),
          )
        ],
      ),
    );
  }
}
// Center(
// Center is a layout widget. It takes a single child and positions it
// in the middle of the parent.
// child: Column(
//   // Column is also a layout widget. It takes a list of children and
//   // arranges them vertically. By default, it sizes itself to fit its
//   // children horizontally, and tries to be as tall as its parent.
//   //
//   // Column has various properties to control how it sizes itself and
//   // how it positions its children. Here we use mainAxisAlignment to
//   // center the children vertically; the main axis here is the vertical
//   // axis because Columns are vertical (the cross axis would be
//   // horizontal).
//   //
//   // TRY THIS: Invoke "debug painting" (choose the "Toggle Debug Paint"
//   // action in the IDE, or press "p" in the console), to see the
//   // wireframe for each widget.
//   mainAxisAlignment: MainAxisAlignment.center,
//   children: <Widget>[
//     OutlinedButton(
//       style: OutlinedButton.styleFrom(
//         primary: Colors.black,
//         side: const BorderSide(
//           color: Colors.black
//         )
//       ),
//         onPressed: (){
//       debugPrint('Click received');
//     }, child: const Text('Click me'))
//   ],
// ),
//       ), // This trailing comma makes auto-formatting nicer for build methods.
//     );
//   }
// }