

int_vec<-function(a){  # vector to int
  n<-length(a)
  value<-sum( a*10^( (n-1):0 )  )
  return( value )
}

sep_change<-function(a){
  return(all(a==9))
}

char_add<-function(x,y){  # add int(x) and int(y), output as vector(int) 
  #x<-char_vec("12399")
  #y<-1
  n<-length(x)
  m<-length(y)
  if(n<m){
    a<-x
    x<-y
    y<-a
    n<-length(x)
    m<-length(y)
  }
  x<-rev(x)
  y<-c(rev(y),rep(0,n-m))
  tmp<-0
  for (i in 1:n) {
    if(x[i]+y[i]+tmp>=10){
      x[i]<-(x[i]+y[i]+tmp)%%10
      tmp<-1
      if(i==n){
        x<-c(x,1)
      }
    }else{
      x[i]<-(x[i]+y[i]+tmp)%%10
      tmp<-0
    }
  }
  return(rev(x))
}

char_minus<-function(x,y){
  # x<-char_vec("100000")
  # y<-1
  n<-length(x)
  m<-length(y)
  x<-rev(x)
  y<-c(rev(y),rep(0,n-m))
  tmp<-0
  for (i in 1:n) {
    if(x[i]-tmp<y[i]){
      x[i]<-(x[i]-tmp-y[i]+10)%%10
      tmp<-1
    }else{
      x[i]<-(x[i]-y[i]-tmp)%%10
      tmp<-0
    }
  }
  x<-rev(x)
  if(x[1]==0){
    return(x[-1])
  }else{
    return(x)
  }
}

char_vec<-function(x){  # char to vector
  result<- strsplit(as.character(x), split = "")
  result<-result[[1]]
  n<-length(result)
  a<-rep(0,n)
  for(i in 1:n){
    a[i]<-as.integer(result[i])
  }
  return(a)
}

calc_loc<-function(x){ # see the digit of x in sequence 1234...(x-1)(x)
  if(x<10){
    return(x)
  }else{
    n<-floor(log10(x))+1
    return(x*n-sum( 10^((n-1):1)-1 ))
  }
  
}

judge<-function(a,begin,n_sep){
  flag=1
  a<-char_vec(a) # change number into vector to avoid losing precision
  n<-length(a)
  tmp<-begin
  check_0<-0
  while (tmp+n_sep<=n) {
    check_now<-a[(tmp+1) :(tmp+n_sep)]
    #the first number can't be zero
    if((check_now[1]==0)){
      flag=0
      break
    }
    if(tmp==begin){
      check_old<- char_minus(check_now,1)    # at first, get the char version of check_now-1
      check_0<-check_now
    }
    if(begin!=0 && tmp==begin){
      if( !all(a[1:begin] == rev(rev(check_old)[1:begin]))     ){ # match the left side, if fail, then break
        flag=0
        break
      }
    }
    print('check_now and check_old:')
    print(check_now)
    print(check_old)
    # sometimes the length of two different vectors are not equal, then break
    # or the two vector with same length are not totally different, then break
    if(  length(char_minus(check_now,1))!= length(check_old)  || !all(char_minus(check_now,1) == check_old)  ){  #check if the new number is behind the old number
      flag=0
      break
    }
    print('check now is:')
    print(check_now)
    print('check old is:')
    print(check_old)
    tmp<-tmp+n_sep
    check_old<-check_now
    n_sep<- n_sep + sep_change(check_now)
    print('tmp:')
    print(tmp)
    print(n_sep)
    
  }
  #how many number remain rightside
  n_right<-n-tmp
  check_now<-  char_add(check_now,1) 
  print('check_now after while:')
  print(check_now)
  print('flag is:')
  print(flag)
  print('nright is:')
  print(n_right)
  # match the right side
  if(n_right!=0&&flag!=0){
    if( !all(a[(n-n_right+1):n] == check_now[1:n_right]) ){ 
      flag=0
    }
  }
  first_complete_num<-int_vec(check_0)
  
  return( ifelse(flag==0,0,calc_loc(first_complete_num-1)-begin+1) )
  
}  

judge2<-function(a,begin,n_sep){

  ##        (xxxx|xxxxx)(xxxxxxx|xx)
  ##length       (begin)(n-begin)( sep-(n-begin) )
  ##number    x1   a1       a2   x2
  ##  x1 * 10^(begin) + a1 + 1 == a2 * 10^(sep-n+begin) + x2
  ##  sep < n => begin > sep-n+begin
  ##          =>  10^(begin) | a2 * 10^(sep-n+begin) + x2 - a1 -1 
  n<-nchar(a)
  a<-char_vec(a)
  n0<-n_sep+begin-n
  a1<-a[1:begin]
  a2<-a[-(1:begin)]
  if(a2[1]==0){
    return(0)
  }
  print('a1:')
  print(a1)
  print('a2')
  print(a2)
  aa2<-vec_minus(c(1,rep(0,n_sep+begin)),c(a2,rep(0,n_sep-n+begin)))
  print('aa2')
  print(aa2)
  print('aa1')
  aa1<-vec_add(a1,1)
  print(aa1)
  aa<-vec_add(aa1,aa2) # = 10^infty - a2 * 10^(sep-n+begin) + a1 + 1 = x2 (mod 10^begin)
  x2<- rev(rev(aa)[1:(begin)])
  print('x2')
  print(x2)
  #remove the zeros on the left so as to do char2int
  while (x2[1]==0&&length(x2)>1) {
    x2<-x2[-1]
  }
  
  #x2<- ( -a2*10^(n_sep-n+begin)+1+a1 )%%10^begin
  
  first_complete_num<-int_vec(a2)*10^(n_sep-n+begin)+int_vec(x2)
  
  #x2 must be smaller than 10^(sep-n+begin)
  return( ifelse(length(x2)>= (1+n_sep-n+begin),0,calc_loc(first_complete_num-1)-begin+1) )
  
}

num_str1<-function(x){

  n<-nchar(x)
  ##### 0000000
  if(all(char_vec(x)==0)){
    begin<- -1
    first_complete_num<- 10^n
    return(  calc_loc(first_complete_num-1)-begin+1     )
  }
  
  a<-judge(x,0,1)
  if(a!=0){
    return(a)
  }
  for (i in 2:n) {
    b<-vector()
    for(j in 0: (i-1)  ){
      if(i+j<=n){
        a<-judge(x,j,i)
      }else{
        a<-judge2(x,j,i)
      }
      b<-c(b,a)
    }
    if(max(b)!=0){
      return(min(b[which(b>0)]))
    }
  }
}

set.seed(3)
a<-as.character(as.integer(runif(100,1,10)))
b<-a
b[100]<-as.integer(a[100]) +1
a<-paste(c(a,b),collapse = "")
num_str(a)

set.seed(3)
a<-as.character(as.integer(runif(100,1,10)))
b<-a
b[100]<-as.integer(a[100]) +1
a<-paste(c(a[2:100],b[1:99]),collapse = "")
num_str(a)




num_str("12")
num_str("21")

num_str("0")#1 00101
num_str("00")#1 00101
num_str("010")#10 0101
num_str("0100")#100 01001
num_str("000")#1 0001001

judge2("1234567890123414",9,13)

num_str("11110011")#100011 100012
calc_loc(100112) # 100111 100112

set.seed(3)
a<-as.character(as.integer(runif(200,1,10)))
a<-paste(a,collapse = "")
num_str(a)
calc_loc(as.integer(a))



num_str("110000000000000000000000000000000000000000000000001")#1 0001001
calc_loc(11000000000000000000000000000000000000000000000000)
num_str("10001000")
calc_loc(10001001)
num_str("8765432198765432198765432198765432")#1 0001001
calc_loc(876543219876543219)
num_str("9876543210123457078067060006067890")
calc_loc(9876543210123457078067060006067890)
num_str("9876543210123")#1 0001001
num_str("001")#1 00101
calc_loc(100)
num_str("98765432101")
calc_loc(98765432101)

num_str("9")
num_str("10")
num_str("99")
num_str("90")
num_str("999") # 899900
calc_loc(899)


num_str("9910")
num_str("99100")
num_str("12345678910")
num_str("99899991000010001")
num_str("131415161")
num_str("1131415161")#907162534
num_str("9")
num_str("10")
num_str("99")
num_str("90")
num_str("999") # 899900
num_str("00")#1 00101
num_str("010")#10 0101
num_str("0100")#100 01001
num_str("000")#1 0001001
num_str("100000000")
num_str("00000000")


num_str("9980")int_
num_str("79980")
num_str("8999800")
num_str("995680")
