import React from 'react'

const GPTChat = ({message}) => {
  return (
    <div className='bg-[#A4123F] text-white py-3 px-6 rounded-2xl max-w-[80%] md:max-w-[50%] w-fit my-2 text-sm'>
      {message}
    </div>
  )
}

export default GPTChat
