
vecToint<-function(vec){ # vector to int
  return(as.integer(paste(as.character(vec),collapse = '')))
}

digit_add<-function(vec){ 
# judge whether digit should judge. 
# During digit looping, when meet 9,99,999.., next number will have one more digit. So loop digit increases 1.
  if((length(unique(vec)) == 1)&(vec[1]==9)){
    return(1)
  }else{
    return(0)
  }
}

vec_add <- function(x, y){  
  # x&y are vectors, add int(x) and int(y) to sum, output as vector(sum)
  # eg: c(9,8) + c(9,9) = c(1,9,7)
  len_x = length(x)
  len_y = length(y)
  if(len_x < len_y){
    x = c(rep(0, len_y - len_x), x)
  }else{
    y = c(rep(0, len_x - len_y), y)
  }
  len_x = length(x)
  len_y = length(y)
  add = c()
  digit_j = 0
  for(i in len_x:1){
    if(x[i] + y[i] + digit_j < 10){
      add[i] = x[i] + y[i] + digit_j
      digit_j = 0
    }else{
      add[i] = x[i] + y[i] + digit_j - 10
      digit_j = 1
    }
  }
  if(digit_j == 1){
    add[len_x + 1] = 1
    return(c(add[len_x + 1], add[1: len_x]))
  }else{
    return(add)
  }
}

vec_minus<-function(x, y){
  # x > y
  # x&y are vectors, calculate int(x) minux int(y) , output as vector
  # eg: c(9,8) - c(1) = c(9,7)
  len_x = length(x)
  len_y = length(y)
  y = c(rep(0, len_x - len_y), y)
  minux = c()
  digit_d = 0
  for(i in len_x : 1){
    if((x[i] - y[i] - digit_d) < 0){
      minux[i] = x[i] - y[i] + 10 - digit_d
      digit_d = 1 
    }else{
      minux[i] = x[i] - y[i] - digit_d
      digit_d = 0
    }
  }
  while(length(minux) > 1 & minux[1] == 0){
    minux = minux[-1]
  }
  return(minux)
}

intTovec = function(int){  
  # integer to vector : 12 -> c(1,2)
  split =  strsplit(as.character(int),'')[[1]]
  return(as.integer(split))
}

charTovec<-function(char){  # char to vector
  result =  strsplit(as.character(char), split = "")[[1]]
  n = length(result)
  vec = c()
  for(i in 1:n){
    vec[i]<-as.integer(result[i])
  }
  return(vec)
}

ri_loca = function(int){ 
  # see the digit location of x in sequence 1234...(x)  ps : location of last digit of x
  digit = length(strsplit(as.character(int),'')[[1]])
  if(digit == 1){
    return(int)
  }else{
    loca = 9 * sum(10^(c(0: (digit-2))) * c(1: (digit - 1))) + digit * (int - 10^(digit - 1) + 1)   
    return(loca)    
  }
}

no_overlen = function(input, begin_index, gap){
  # This function is to output location of digits of input number when (begin_index + separations) <= length of s.
  # sign means whether the initial number of s is found 
  sign = 1
  input = intTovec(input) # transform integer into vector 
  len = length(input)
  index = begin_index
  initial = 0
  #print('index:')
  #print(index)
  #print(gap)
  while((index+gap) <= len){
    curr_val = input[(index+1): (index+gap)]
    if((curr_val[1]==0)){
      sign=0
      break
    }
    #print(curr_val)
    if(index == begin_index){
      former_val = vec_minus(curr_val,1)   
      initial = curr_val # mark initial loop value.
    }
    #print('curr_val and former_val:')
    #print(curr_val)
    #print(former_val)
    if(begin_index!=0 & index == begin_index){
      # fill the left incomplete number to see whether equal to former number.
      if( sum(input[1:begin_index] == rev(rev(former_val)[1:begin_index])) != begin_index){ 
        sign = 0
        break
      }
    }
    #print('curr_val and former_val:')
    #print(curr_val)
    #print(former_val)
    # When the length of two different vectors are not equal, then break
    # or the two vector with same length are not totally different, then break
    if(  length(vec_minus(curr_val,1))!= length(former_val)  || !all(vec_minus(curr_val,1) == former_val)  ){  #check if the new number is behind the old number
      sign=0
      break
    }
    #print('curr_val is:')
    #print(curr_val)
    #print('former_val is:')
    #print(former_val)
    
    former_val = curr_val # transfer former value to the one behind for next loop.
    index = index + gap
    gap = gap + digit_add(curr_val)
    #print('index:')
    #print(index)
   # print(gap)
  }
  
  # dealing with left right numbers 
  leftright = len - index
  curr_val = vec_add(curr_val,1) 
  #print('curr_val after while:')
  #print(curr_val)
  #print('sign is:')
  #print(sign)
  #print('leftright is:')
  #print(leftright)
  # match the leftright numbers
  if(leftright != 0 & sign != 0){
    if( sum(input[(len - leftright + 1) : len] == curr_val[1:leftright]) != leftright){ 
      sign=0
    }
  }
  initialval = vecToint(initial) # transfer vector initial to integer form
  if(sign == 0){
    return(0)
  }else{
    #print(initialval)
    digit_seq = ri_loca(initialval - 1)
    # Since we find a complete number in s, we should find last digit of former value minus existing part.
    digit_seq = digit_seq - begin_index + 1 
    return(digit_seq)
  }
}  

######
overlen = function(input, begin_index, gap){
  len = nchar(input)
  input = intTovec(input)
  lenover = gap + begin_index - len
  former = input[1:begin_index]
  later = input[-(1:begin_index)]
  if(later[1]==0){
    return(0)
  }
  #print(input[(begin_index + 1): gap])
  #print(former)
  if(all(former==9)){
    fill_former = (vecToint(input[(begin_index + 1): gap])-1) * 10^begin_index 
  }else{
    fill_former = vecToint(input[(begin_index + 1): gap]) * 10^begin_index
  }
  #print(fill_former)
  former_all = fill_former + vecToint(former)
  fill_later = vecToint(input[(begin_index - lenover + 1): begin_index]) + 1
  #print(fill_later)
  if(fill_later == 10^(lenover)){
    later_all = fill_later * vecToint(later)
  }else{
    later_all = vecToint(later) * 10^(lenover) + fill_later
  }
  #print(former_all)
  #print(later_all)
 
  
  initialval = later_all
  
  #x2 must be smaller than 10^(lenover)
  if((later_all - former_all) != 1){
    return(0)
  }else{
    digit_seq = ri_loca(initialval - 1)
    # Since we find a complete number in s, we should find last digit of former value minus existing part.
    digit_seq = digit_seq - begin_index + 1 
    return(digit_seq)
  }
}

num_str = function(s){
  options(scipen=999)
  a = strsplit(s,'')[[1]]
  db = as.double(a)
  l = length(a)
  
  # 00000
  if(length(unique(db))==1 & db[1]==0){ 
    seq = 0
    ll = l
    while(ll>=1){
      seq = seq + 9 * 10^(ll-1) * ll
      ll = ll - 1
    }
    return(seq + 2)
  }
  
  n_sign <- no_overlen(s,0,1)
  if(n_sign != 0){
    return(n_sign)
  }
  for (i in 2 : l) {
    result  = c()
    for(j in 0 : (i-1)){
      if((i+j) <= l){
        num = no_overlen(s, j, i)
      }else{
        num = overlen(s, j, i)
      }
      result = c(result, num)
    }
    if(max(result) != 0){
      output = result[result>0]
      return(min(output))
    }
  }
}