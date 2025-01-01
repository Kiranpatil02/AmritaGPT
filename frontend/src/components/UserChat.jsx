import React from 'react'
import { useState, useEffect } from 'react';
import GPTChat from './GPTChat';


const UserChat = ({message,duration}) => {

  const url="<Backend_url>"


  const [response, setResponse] = useState("");
  const [loading,setloading]=useState(false);
  const query=message
  useEffect(() => {
    const fetchResponse = async () => {
      setloading(true);
      duration(true);
      try {
        const res = await fetch(`${url}get-response/`,{
          method:"POST",
          headers:{
            "Content-Type":"application/json",
          },
          body:JSON.stringify({query}),
        });
        const data=await res.json();
        setResponse(data.response); 
      } catch (error) {
        console.error('Error fetching response:', error);
        setResponse('Failed to fetch response.');
      }
      finally{
        setloading(false);
        duration(false);
      }
    };

    fetchResponse();
  }, [query]);

  return (
    <>
    <div className='bg-white py-3 px-6 rounded-2xl max-w-[65%] w-fit ml-auto my-2 text-sm '>
      {message}
    </div>
    <div>
      {
        loading?(<p className='bg-[#A4123F] text-white py-3 px-6 rounded-2xl max-w-[50%] w-fit my-2 mx-5'>Loading...</p>):
        (    <GPTChat message={response}/>)
      }

    </div>
    </>
  )
}

export default UserChat
