// import {linspace} from '@tensorflow/tfjs';
// const tf = require('@tensorflow/tfjs')

//function to generate an integer in the specified range

function randomInt(min, max) {
    return Math.round(Math.random() * (max - min)) + min;
}

//generate a random float between 0 to 1 
function randomFloat(){
    return Math.random()
}

//this function is uesd to generate a random array and this array is input array to a node
function randomArray(startNum,endNum){
    let arr = []
    let arrLength = randomInt(1,endNum - startNum + 1)
    for(var i = 0; i < arrLength; i++){
        arr.push(randomInt(startNum,endNum))
    }
    
    //sort and remove duplication (if use array.sort() directly,this method cannot sort numbers above 10 )
    arr = [...new Set(arr.sort((a,b) => {
        return a - b
    }))]
    return arr

}
//test
// console.log(randomArray(1,10));

let activationFun = ['sigmoid', 'gaussian', 'sine','cosine','tan','tanh']
function sigmoid(x){
    //Math.exp(x) == e^x
    return 1 / (1 + Math.exp(-x))
}
function gaussian(x,a){
    return (1/(a*Math.sqrt(2*Math.PI))) * Math.exp(-((x*x)/(2*a*a)))
}
function sine(x){
    return Math.sin(x)
}
function cosine(x){
    return Math.cos(x)
}
function tan(x){
    return Math.tan(x)
}
function tanh(x){
    return Math.tanh(x)
}

//the length of cppn can be set in a specific range(min, max) (include ['identity',['x'],[1]], ['identity',['y'],[1]])
function create_random_cppn(min,max){
    
    //cppn: [[activation function,[input array],weight]]
    var cppnArrayLen = randomInt(min,max);
    var cppnArray = new Array(cppnArrayLen);

    //the first two items of outer array are special, so I list them separately
    //for inner array
    //the first item of array is choose a random activation function for this node,
    //the second item of array is input nodes array,
    //The third item should be an *array of weights*, one weight for each input. (between 0 to 1)
    cppnArray[0] = ['identity',['x'],[1]]
    cppnArray[1] = ['identity',['y'],[1]]
    for(var i = 2; i < cppnArrayLen; i++){
        var randomArr = randomArray(0,i-1)
        var weightList = []
        for(var j = 0; j < randomArr.length; j++){
            weightList.push(randomFloat())
        }
        cppnArray[i] = [activationFun[randomInt(0,3)],randomArr,weightList]
    }
    //change the last node activation function to 'sigmoid
    cppnArray[cppnArrayLen-1].splice(0,1,'sigmoid')
    return cppnArray
}
// test
// var random_cppn = create_random_cppn(10,15);
// console.log(random_cppn);

function run_cppn(init_cppn,x,y){
    //to get the length of the cppn
    var init_cppnLength = init_cppn.length
    //create a output list to every node's output
    var outputList = new Array(init_cppnLength);
    //because the first and second items' output of cppp are special
    //we set directly
    outputList[0] = x
    outputList[1] = y
    for(var i = 2; i < init_cppnLength; i++){
        // calculate the weighted sum of inputs
        var weightedSum = 0
        //input nodes array has the same length as weight array:
        //example (one of node in cppn): arr = [ 'gaussian', [ 0, 1 ], [ 0.8201723532981295, 0.4148916826880116 ] ]
        //arr[1].length == arr[2].length
        for(var j = 0; j < init_cppn[i][1].length; j++){
            weightedSum += outputList[init_cppn[i][1][j]] * init_cppn[i][2][j]
        }

        // calcuate every node output
        if(init_cppn[i][0] == 'sigmoid'){
            outputList[i] = sigmoid(weightedSum)
        }else if(init_cppn[i][0] == 'gaussian'){
            outputList[i] = gaussian(weightedSum,3)
        }else if(init_cppn[i][0] == 'sine'){
            outputList[i] = sine(weightedSum)
        }else if(init_cppn[i][0] == 'cosine'){
            outputList[i] = cosine(weightedSum)
        }else if(init_cppn[i][0] == 'tan'){
            outputList[i] = tan(weightedSum)
        }else{
            outputList[i] = tanh(weightedSum)
        }
    }
    //By default we set the output of the last node as the output of the entire cppn
    var run_cppn_output = outputList[init_cppnLength-1]
    return run_cppn_output
}
//test
// var aaa = create_random_cppn(5,10)
// var run_a_cppn = run_cppn(aaa,80,90)
// console.log(run_a_cppn);


// check_cppn is used to check cppn that 
// if every item in input array (inner cppn array) is less than the index of the node;
//check if every node has at least one input
function check_cppn(cppnArray){
    // console.log(cppnArray)
    var cppnlength = cppnArray.length
    for(var i = 2; i < cppnlength; i++){
        for(var j = 0; j < cppnArray[i][1].length; j++)
        if( cppnArray[i][1][j] > i || cppnArray[i][1][j] == i || cppnArray[i][1].length == 0){
            console.log('problem index: ' + i)
            return false
        }
    }
    return true
}
// var aCppn = create_random_cppn()
// console.log(check_cppn(aCppn))


//1.mutate by changing activation function
//here change one activation function in one of nodes.
function cppn_mutate_change_activationFun(cppnArray){
    // console.log(cppnArray)
    var cppnlength = cppnArray.length
    

    //deep copy from activation function array

    // ------------------------------------------------------
    var [...activationFun1] = activationFun

    //change the activation function: 
    // randomIntNum is the index of item which activation function will be changed
    var randomIntNum = randomInt(2,cppnlength-2)
    // console.log(randomIntNum)

    //choose a random item of outer array, its first item is activation function
    var originalAcFun = cppnArray[randomIntNum][0]

    // activationFun1 = ['sigmoid', 'gaussian', 'sine', 'linear']
    // here if this node activation function is 'sine', 
    // we delete 'sine' from activationFun1 and choose another activation function randomly.
    activationFun1.splice(activationFun1.indexOf(originalAcFun),1)

    //choose a new random activation function
    var newAcFun = activationFun1[randomInt(0,2)]
    
    //The new activation function will replace the old one
    cppnArray[randomIntNum].splice(0,1,newAcFun)
    cppnArray[cppnArray.length-1].splice(0,1,'sigmoid')
    // console.log('m1',cppnArray);
    return cppnArray
}
//test
// var aCppn = create_random_cppn()
// console.log(mutate1_cppn(aCppn))


//2. mutate edge
//change a edge(change an item in input array) in one node's input array
function cppn_mutate_change_edge(cppnArray){
    // console.log(111,cppnArray,111)
    var cppnlength = cppnArray.length

    // randomIntNum is the index of item which edge will be changed
    // var randomIntNum = randomInt(2,cppnlength-1)
    // console.log('index of being changed node: ' + randomIntNum)

    // if(randomIntNum == cppnArray[randomIntNum][1].length){
    //     randomIntNum = randomInt(2,cppnlength-1)
    // }
    function generateAInt(cppnlength){
        var randomIntNum = randomInt(2,cppnlength-1)
        if(randomIntNum !== cppnArray[randomIntNum][1].length){
            return randomIntNum
        }else{
            return generateAInt(cppnlength)
        }
    }
    var randomIntNum = generateAInt(cppnlength)
    //change one of items in the input array
    // var inputArr = cppnArray[randomIntNum][1]
    var inputArr = cppnArray[randomIntNum][1]

    var inputArrLength = inputArr.length

    //choose a random edge in input array to change, randomIntEdge is index which need to be changed
    var randomEdgeIndex = randomInt(0,inputArrLength - 1)
    
    

    //inputArr.indexOf(randomNewEdge): check if input array has new edge, if doesn't have, it will return -1 (if it has, it will return index)
    // if (inputArr.indexOf(randomNewEdge) == -1 ){
    //     inputArr.splice(randomEdgeIndex,1,randomNewEdge)
    // }
    
    function generateNewEdge(inputArr){
        
        //generate a new edge (input), this edge should less than current node index
        var randomNewEdge = randomInt(0,randomIntNum - 1)
        // console.log(inputArr.indexOf(randomNewEdge));
        //inputArr.indexOf(randomNewEdge): check if input array has new edge, if doesn't have, it will return -1 (if it has, it will return index)
        if ( inputArr.indexOf(randomNewEdge) == -1 ){
            inputArr.splice(randomEdgeIndex,1,randomNewEdge)
            inputArr = inputArr.sort()
            
            return inputArr
        }else{
            return generateNewEdge(inputArr)
        }
    }


    var newInputArr = generateNewEdge(inputArr)
    // console.log(333,newInputArr);
    cppnArray[randomIntNum].splice(1,1,newInputArr)
    cppnArray[cppnArray.length-1].splice(0,1,'sigmoid')

    // console.log('m2',cppnArray);
    return cppnArray
}
// test
// var aCppn = create_random_cppn()
// console.log(cppn_mutate_change_edge(aCppn))


//3. mutate a node (add a new node)
function cppn_mutate_add_node(cppnArray){
    // console.log(cppnArray)
    var cppnLength = cppnArray.length

    // randomIntNum is the index that we will insert a new node
    var randomIntNum = randomInt(2,cppnLength)
    // console.log(randomIntNum)
    var randomArr = randomArray(0,randomIntNum-1)
    var randomWeightArr = []
    for(var j = 0; j < randomArr.length; j++){
        randomWeightArr.push(randomFloat())
    }
    //generate a new node
    var newNode = [activationFun[randomInt(0,3)],randomArr,randomWeightArr]


    //insert the new node to cppn
    cppnArray.splice(randomIntNum,0,newNode)


    //because we add a new node, so don't need to  check
    //every node input array (edge array)
    // console.log(cppnArray)
    return cppnArray
}
// var aCppn = create_random_cppn()
// mutate3_cppn(aCppn)

//4. delete a Node
function cppn_mutate_delete_node(cppnArray){
    // console.log(cppnArray)
    var cppnLength = cppnArray.length

    // randomIntNum is the index of the node that we will delete
    var randomIntNum = randomInt(2,cppnLength)
    // console.log(randomIntNum);
    cppnArray.splice(randomIntNum,1)
    var newCppnLength = cppnArray.length
    //one of the node have been deleted, 
    //we should check input array in inner array of newCppn,
    //check if the items of input array is less than the index of the node
    //for example: here is a node, suppose the index of the node is 6,
    // in [4,6] item 6 is >= index 6 so we need delete this item and delete its weight as well
    // [ 'gaussian', [ 4, 6 ], [ 0.5462298339693481, 0.40477407398019394 ] ]

    // example:
    //here i already do some operation to fix the problem: 
    // originally, node 5 has input from node 4, after mutation,
    //node 5 becomes node 4, so i delete related input item in input array
    // [
    //     [ 'identity', [ 'x' ], [ 1 ] ],
    //     [ 'identity', [ 'y' ], [ 1 ] ],
    //     [ 'sigmoid', [ 0 ], [ 0.533737270267691 ] ],
    //     [ 'gaussian', [ 1 ], [ 0.8646332971156734 ] ],
    //     [ 'cosine', [ 2, 3 ], [ 0.3258700921556399, 0.8032051350232565 ] ],  this node (node 4) will be deleted
    //     [
    //       'sine',
    //       [ 1, 2, 4 ],                                                     this node (node 5) will become node 4
    //       [ 0.23014072524043216, 0.9277548002857761, 0.6218220332012201 ]
    //     ]
    //   ]
    //   4     this is index of node will be deleted
    //   [
    //     [ 'identity', [ 'x' ], [ 1 ] ],
    //     [ 'identity', [ 'y' ], [ 1 ] ],
    //     [ 'sigmoid', [ 0 ], [ 0.533737270267691 ] ],
    //     [ 'gaussian', [ 1 ], [ 0.8646332971156734 ] ],
    //     [ 'sine', [ 1, 2 ], [ 0.23014072524043216, 0.9277548002857761 ] ]  this node (new node 4, originally it is node 5)
    //   ]
    for(var i = randomIntNum; i < newCppnLength; i++){
        for(var j = cppnArray[i][1].length - 1; j >= 0; j--)
            if( cppnArray[i][1][j] > i || cppnArray[i][1][j] == i){
                cppnArray[i][1].splice(j,1)
                cppnArray[i][2].splice(j,1)
        }
    }
    if(check_cppn(cppnArray)){
        
        cppnArray[cppnArray.length-1].splice(0,1,'sigmoid')
        // console.log('m4',cppnArray);
        return cppnArray
    }

}
// test
// var aCppn = create_random_cppn(5,8)
// var mutateCppn = cppn_mutate_delete_node(aCppn)
// console.log(mutateCppn);


//5. change weight
function cppn_mutate_change_weight(cppnArray){
    // console.log(cppnArray)
    var cppnlength = cppnArray.length


    // randomIntNum is the index of item (node) which weight will be changed
    var randomIntNum = randomInt(2,cppnlength-1)
    // console.log(randomIntNum)

    var weightLength = cppnArray[randomIntNum][2].length

    //choose a random item in weight list to change
    var randomIndex = randomInt(0,weightLength-1)
    

    cppnArray[randomIntNum][2].splice(randomIndex,1,randomFloat())
    // console.log(cppnArray)
    // console.log('m5',cppnArray);
    return cppnArray
}
// test
// var aCppn = create_random_cppn(5,10)
// cppn_mutate_change_weight(aCppn)

//"roulette wheel" for choose a random mutation function
function rouletteWheel(cppn){
    //p1:change_activationFun; p2:change_edge; p3:add_node; p4:delete_node; p5:change_weight
    var p1 = 0.2
    var p2 = 0.2
    var p3 = 0.2
    var p4 = 0.2
    var p5 = 0.2

    var x = Math.random()
    if(x <= p1) {
        var newCppn_change_activationFun = cppn_mutate_change_activationFun(cppn)
        return newCppn_change_activationFun
    }else if(x > p1 && x <= p1 + p2){
        var newCppn_change_edge = cppn_mutate_change_edge(cppn)
        return newCppn_change_edge
    }else if(x > p1 + p2 &&  x <= p1 + p2 + p3){
        var newCppn_add_node = cppn_mutate_add_node(cppn)
        return newCppn_add_node
    }else if(x > p1 + p2 + p3 && x <= p1 + p2 + p3 + p4){
        var newCppn_delete_node = cppn_mutate_delete_node(cppn)
        return newCppn_delete_node
    }else if(x > p1 + p2 + p3 + p4  && x <= p1 + p2 + p3 + p4 + p5){
        var newCppn_change_weight = cppn_mutate_change_weight(cppn)
        return newCppn_change_weight
    }

    
}

function run_cppn_grid(cppn, xmin, xmax, xsize, ymin, ymax, ysize){
    //here i create a tensor and convert a tensor to regular js array

    //here is example: tf.linspace(0,9,10), the range is [0,9], 
    // the length of tensor is 10,so output is [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
    const xtensor = tf.linspace(xmin, xmax, xsize)
    const ytensor = tf.linspace(ymin, ymax, ysize)

    //convert tensor to regular js array
    const xArray = Array.from(xtensor.dataSync())
    const yArray = Array.from(ytensor.dataSync())

    var cppn_grid_array = []
    xArray.forEach((xitem,xindex) => {
        yArray.forEach((yitem,yindex)=> {
            var cppnX = run_cppn(cppn,xitem,yitem)
            cppn_grid_array.push(cppnX)
        })
    })
    
    
    return cppn_grid_array
}
// var initCppn = create_random_cppn(15,30)
// console.log(check_cppn(initCppn));
// var ccc = run_cppn_grid(initCppn,0,1,10,0,1,10)
// console.log(ccc);


//generate 2d array
function run_cppn_grid2(cppn, xmin, xmax, xsize, ymin, ymax, ysize){
    //here i create a tensor and convert a tensor to regular js array

    //here is example: tf.linspace(0,9,10), the range is [0,9], 
    // the length of tensor is 10,so output is [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
    const xtensor = tf.linspace(xmin, xmax, xsize);
    const ytensor = tf.linspace(ymin, ymax, ysize);

    //convert tensor to regular js array
    const xArray = Array.from(xtensor.dataSync());
    const yArray = Array.from(ytensor.dataSync());
    
    var cppn_grid_array = [];
    // var cppn_grid_array1 = []
    var innerArray = [];
    yArray.forEach((yitem,yindex) => {
        xArray.forEach((xitem,xindex)=> {
            var cppnX = run_cppn(cppn,xitem,yitem);
            // cppn_grid_array1.push(cppnX)
            if(innerArray.length < xsize ){
                innerArray.push(cppnX);
            }else{
                cppn_grid_array.push(innerArray);
                innerArray = [];
                innerArray.push(cppnX);
            }
            
        })
        if(yindex == ysize-1){
            cppn_grid_array.push(innerArray);
        }
    })
    return cppn_grid_array
}
// var initCppn = create_random_cppn(15,30);
// var ccc = run_cppn_grid2(initCppn,0,1,10,0,1,10);
// console.log(ccc);



// ----------------------------------------------------------------------------------

//here are two function to output 3 number for rgb
function run_cppn2(init_cppn,x,y){
    //to get the length of the cppn
    var init_cppnLength = init_cppn.length
    //create a output list to every node's output
    var outputList = new Array(init_cppnLength);
    //because the first and second items' output of cppp are special
    //we set directly
    outputList[0] = x
    outputList[1] = y
    for(var i = 2; i < init_cppnLength; i++){
        // calculate the weighted sum of inputs
        var weightedSum = 0
        //input nodes array has the same length as weight array:
        //example (one of node in cppn): arr = [ 'gaussian', [ 0, 1 ], [ 0.8201723532981295, 0.4148916826880116 ] ]
        //arr[1].length == arr[2].length
        for(var j = 0; j < init_cppn[i][1].length; j++){
            weightedSum += outputList[init_cppn[i][1][j]] * init_cppn[i][2][j]
        }

        // calcuate every node output
        if(init_cppn[i][0] == 'sigmoid'){
            outputList[i] = sigmoid(weightedSum)
        }else if(init_cppn[i][0] == 'gaussian'){
            outputList[i] = gaussian(weightedSum,3)
        }else if(init_cppn[i][0] == 'sine'){
            outputList[i] = sine(weightedSum)
        }else if(init_cppn[i][0] == 'cosine'){
            outputList[i] = cosine(weightedSum)
        }else if(init_cppn[i][0] == 'tan'){
            outputList[i] = tan(weightedSum)
        }else{
            outputList[i] = tanh(weightedSum)
        }
    }
    var run_cppn_output = []
    run_cppn_output.push(outputList[init_cppnLength-3])
    run_cppn_output.push(outputList[init_cppnLength-2])
    run_cppn_output.push(outputList[init_cppnLength-1])
    return run_cppn_output
}
// return 3 number from every cppn
function run_cppn_grid3(cppn, xmin, xmax, xsize, ymin, ymax, ysize){
    const xtensor = tf.linspace(xmin, xmax, xsize);
    const ytensor = tf.linspace(ymin, ymax, ysize);
    const xArray = Array.from(xtensor.dataSync());
    const yArray = Array.from(ytensor.dataSync());
    var cppn_grid_array = [];
    
    var innerArray = [];
    yArray.forEach((yitem,yindex) => {
        xArray.forEach((xitem,xindex)=> {
            var cppnX = run_cppn2(cppn,xitem,yitem);
            // cppn_grid_array1.push(cppnX)
            if(innerArray.length < xsize ){
                innerArray.push(cppnX);
            }else{
                cppn_grid_array.push(innerArray);
                innerArray = [];
                innerArray.push(cppnX);
            }
        })
        if(yindex == ysize-1){
            cppn_grid_array.push(innerArray);
        }
    })
    return cppn_grid_array
}